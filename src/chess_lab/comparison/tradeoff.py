"""Depth-accuracy profiler.

Measures how deep a bot searches in a fixed time per move and how often it finds the best move.

The central trade-off of the project is that a slower, more accurate evaluation reaches a lower
depth in the same time per move. :func:`profile` measures both sides for one player.
"""

import logging
import math
import multiprocessing
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess
import pandas as pd

from chess_engine import _core
from chess_engine.bots.search_bot import SearchBot
from chess_engine.bots.time_manager import TimeManager
from chess_engine.models import split_spec
from chess_lab.models.cards import card_path, update_card
from chess_lab.models.definitions import get_model_class

logger = logging.getLogger(__name__)

SINGLE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)
HEURISTICS = ("h1",)


def fixed_time_manager(seconds: float) -> TimeManager:
    """Time manager whose per-move budget is exactly ``seconds`` (no extension, no margin).

    :param seconds: Seconds allowed for each move.
    :return: The time manager.
    """
    return TimeManager(
        move_fraction=1.0,
        min_turn_time=seconds,
        max_turn_time=seconds,
        max_clock_share=1.0,
        extension=1.0,
        safety_margin=0.0,
    )


def split_techniques(spec: str) -> Tuple[str, Tuple[str, ...]]:
    """Split ``name+tech1,tech2`` into the player and the techniques of selective search.

    ``name+all`` switches every technique on. A spec without ``+`` has no techniques.

    :param spec: Player specification, possibly with a ``+`` suffix.
    :return: ``(player specification, techniques)``.
    """
    base, _, rest = spec.partition("+")
    if not rest:
        return base, ()
    names = list(_core.TECHNIQUES) if rest == "all" else [n for n in rest.split(",") if n]
    return base, tuple(names)


def make_player(
    spec: str,
    time_manager: TimeManager,
    max_depth: int = 30,
    **search_options: Any,
) -> SearchBot:
    """Build an iterative-deepening player for measurements: no opening book and no tablebases.

    :param spec: ``"h1"`` for a hand-crafted heuristic, a registered model
        name or ``name@tag`` (its player is named after the spec). A ``+`` suffix switches techniques of selective search on (``linear_pst+pvs,lmr``
        or ``linear_pst+all``); without it the player uses the reference search, so that measurements
        compare evaluations alone.
    :param time_manager: Time budgeting policy.
    :param max_depth: Deepest iteration to run.
    :param search_options: Other options of the bot (``estimator``, ``min_depth``...); they may
        switch the book (``opening_book``) or the tables (``tablebase``) on.
    :return: The player.
    """
    search_options = {"opening_book": None, "tablebase": None, **search_options}
    label = spec
    techniques: Tuple[str, ...] = ()  # the reference search, unless the spec asks for techniques
    if "+" in spec:
        spec, techniques = split_techniques(spec)
    search_options.setdefault("techniques", techniques)
    return SearchBot(
        spec, name=label, max_depth=max_depth, time_manager=time_manager, **search_options
    )


def _summary(rows: List[Tuple[int, int, float, Optional[bool]]]) -> Dict[str, float]:
    """Aggregate per-position ``(depth, nodes, seconds, agrees)`` measurements.

    :param rows: Measurements; ``agrees`` is None when the position has no reference move.
    :return: Mean / median depth, nodes per second, search agreement and position count.
    """
    if not rows:
        return {"positions": 0}
    depths = [r[0] for r in rows]
    judged = [r[3] for r in rows if r[3] is not None]
    return {
        "positions": len(rows),
        "mean_depth": statistics.fmean(depths),
        "median_depth": statistics.median(depths),
        "nodes_per_second": sum(r[1] for r in rows) / max(sum(r[2] for r in rows), 1e-9),
        "search_agreement": sum(judged) / len(judged) if judged else float("nan"),
    }


def profile(spec: str, positions: pd.DataFrame, seconds: float) -> Dict[str, object]:
    """Search every position for ``seconds`` and report depth, speed and agreement.

    :param spec: Player specification, see :func:`make_player`.
    :param positions: Frame with ``fen``, ``best_move`` (UCI or None) and ``quiet`` columns.
    :param seconds: Seconds per move.
    :return: ``positions`` plus ``overall`` / ``quiet`` / ``non_quiet`` summaries.
    """
    player = make_player(spec, fixed_time_manager(seconds))
    rows = []
    for fen, best_move, quiet in zip(
        positions["fen"], positions["best_move"], positions["quiet"], strict=True
    ):
        board = chess.Board(fen)
        player.on_game_start(board.turn, "profiler")  # fresh transposition table per position
        start = time.perf_counter()
        move = player.get_move(board, seconds)
        elapsed = time.perf_counter() - start
        stats = player.last_search_stats
        agrees = None if best_move is None or pd.isna(best_move) else move.uci() == best_move
        rows.append((stats.completed_depth, stats.nodes_visited, elapsed, agrees, bool(quiet)))
    return {
        "positions": len(rows),
        "overall": _summary([r[:4] for r in rows]),
        "quiet": _summary([r[:4] for r in rows if r[4]]),
        "non_quiet": _summary([r[:4] for r in rows if not r[4]]),
    }


def _profile_job(job: Tuple[str, float, pd.DataFrame]) -> Optional[Dict[str, object]]:
    """Profile one ``(player, seconds)`` pair; None if the model has no saved artifact."""
    spec, seconds, positions = job
    try:
        result = profile(spec, positions, seconds)
    except FileNotFoundError:
        return None
    return {"player": spec, "seconds": seconds, **result}


def validate_specs(specs: Sequence[str]) -> None:
    """Fail fast on a player that is neither a heuristic nor a registered model.

    :param specs: Player specifications (``name`` or ``name@tag``).
    :raises ValueError: If a spec is unknown.
    """
    for spec in specs:
        spec = split_techniques(spec)[0]
        if spec in HEURISTICS:
            continue
        try:
            get_model_class(split_spec(spec)[0])
        except (KeyError, ValueError) as error:
            raise ValueError(f"unknown player {spec!r}: {error.args[0]}") from error


def run_profiles(
    specs: Sequence[str], seconds: Sequence[float], positions: pd.DataFrame, parallel: int = 1
) -> Dict[str, List]:
    """Profile every player at every time, optionally one player/time per process.

    With ``parallel > 1`` every worker is limited to one thread (BLAS, TensorFlow) so that the
    workers do not compete for cores; published numbers should still use ``parallel=1``.

    :param specs: Player specifications.
    :param seconds: Seconds per move to try.
    :param positions: Positions to search, see :func:`profile`.
    :param parallel: Number of worker processes (1 runs in this process).
    :return: ``results`` (one per profiled pair) and ``skipped`` (players without a saved model).
    :raises ValueError: If a spec is unknown (nothing is profiled then).
    """
    validate_specs(specs)
    jobs = [(spec, secs, positions) for spec in specs for secs in seconds]
    if parallel > 1:
        saved = {name: os.environ.get(name) for name in SINGLE_THREAD_VARIABLES}
        os.environ.update(dict.fromkeys(SINGLE_THREAD_VARIABLES, "1"))
        try:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=parallel, mp_context=context) as pool:
                outcomes = list(pool.map(_profile_job, jobs))
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    else:
        outcomes = [_profile_job(job) for job in jobs]
    skipped = []
    for (spec, _, _), outcome in zip(jobs, outcomes, strict=True):
        if outcome is None and spec not in skipped:
            logger.warning("skipping %s: no saved model artifact", spec)
            skipped.append(spec)
    return {"results": [o for o in outcomes if o is not None], "skipped": skipped}


def record_cards(results: Sequence[Dict[str, Any]]) -> List[str]:
    """Write the ``tradeoff`` section of the model cards of the profiled players.

    Each section maps the seconds per move (as in ``"1"``, ``"0.5"``) to the mean completed depth
    and the overall search agreement of the player. Heuristic players and models without a card
    are skipped, as are results without any measured position.

    :param results: Output of :func:`run_profiles`.
    :return: The players whose card was updated.
    """
    recorded: List[str] = []
    for result in results:
        spec, overall = str(result["player"]), result["overall"]
        if spec in HEURISTICS or "mean_depth" not in overall:
            continue
        if not card_path(spec).exists():
            logger.warning("no model card for %s: tradeoff section not written", spec)
            continue
        agreement = overall["search_agreement"]
        if isinstance(agreement, float) and math.isnan(agreement):  # no position judged
            agreement = None
        seconds = f"{float(result['seconds']):g}"
        update_card(
            spec,
            tradeoff={
                seconds: {"mean_depth": overall["mean_depth"], "search_agreement": agreement}
            },
        )
        if spec not in recorded:
            recorded.append(spec)
    return recorded
