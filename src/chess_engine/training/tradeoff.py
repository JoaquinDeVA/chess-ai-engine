"""Depth-accuracy profiler.

Measures how deep a bot searches in a fixed time per move and how often it finds the best move.

The central trade-off of the project is that a slower, more accurate evaluation reaches a lower
depth in the same time per move. :func:`profile` measures both sides for one player.
"""

import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple, Union

import chess
import pandas as pd

from chess_engine.bots._search_bots import IterativeDeepeningBot
from chess_engine.bots.iterative_heuristic_bot import IterativeHeuristicBot
from chess_engine.bots.iterative_model_book_bot import IterativeModelBookBot
from chess_engine.evaluation.heuristics import h2, h3
from chess_engine.knowledge.opening_book import OpeningBook
from chess_engine.models import EvaluationModel
from chess_engine.search.time_management import TimeManager

HEURISTICS = {"h2": h2, "h3": h3}


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


def make_player(
    spec: Union[str, EvaluationModel], time_manager: TimeManager, max_depth: int = 30
) -> IterativeDeepeningBot:
    """Build an iterative-deepening player without an opening book.

    :param spec: ``"h2"`` / ``"h3"`` for a hand-crafted heuristic, any other string for a
        registered model name, or an :class:`EvaluationModel` instance.
    :param time_manager: Time budgeting policy.
    :param max_depth: Deepest iteration to run.
    :return: The player.
    """
    if isinstance(spec, str) and spec in HEURISTICS:
        return IterativeHeuristicBot(
            eval_function=HEURISTICS[spec], max_depth=max_depth, time_manager=time_manager
        )
    return IterativeModelBookBot(
        model=spec, opening_book=OpeningBook({}), max_depth=max_depth, time_manager=time_manager
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


def profile(
    spec: Union[str, EvaluationModel], positions: pd.DataFrame, seconds: float
) -> Dict[str, object]:
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


def run_profiles(
    specs: Sequence[str], seconds: Sequence[float], positions: pd.DataFrame, parallel: int = 1
) -> List[Dict[str, object]]:
    """Profile every player at every time, optionally one player/time per process.

    :param specs: Player specifications.
    :param seconds: Seconds per move to try.
    :param positions: Positions to search, see :func:`profile`.
    :param parallel: Number of worker processes (1 runs in this process).
    :return: One result per profiled pair; players without a saved model are left out.
    """
    jobs = [(spec, secs, positions) for spec in specs for secs in seconds]
    if parallel > 1:
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            results = list(pool.map(_profile_job, jobs))
    else:
        results = [_profile_job(job) for job in jobs]
    return [r for r in results if r is not None]
