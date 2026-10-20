"""Parallel round-robin tournament between players at equal time per move.

Every unordered pair of players meets on every opening twice, once with each colour. Games run
in a process pool and every worker builds its own players from the specification strings (bots
are not picklable). Each finished game is appended to a JSON-lines file, so an interrupted run
resumes where it stopped.
"""

import itertools
import json
import logging
import math
import multiprocessing
import os
import random
import re
import statistics
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import chess

from chess_engine.bots.bot_registry import BotRegistry
from chess_engine.comparison.tradeoff import HEURISTICS, make_player
from chess_engine.engine.game_manager import FORFEIT_TERMINATIONS, GameManager
from chess_engine.models.cards import card_path, load_card, save_card
from chess_engine.search.time_management import TimeManager

logger = logging.getLogger(__name__)

#: Player whose rating is fixed at 0 Elo.
ANCHOR = "h1"

#: Scores are clipped to ``[CLIP, 1 - CLIP]`` before taking a logarithm.
CLIP = 0.01

#: Pools that may be rebuilt in a row, with no game finished in between, before the run gives up.
MAX_POOL_REBUILDS = 5

#: Broken pools a game may be in flight for before it is recorded as an error.
MAX_DEATHS_PER_GAME = 2

#: ``(white spec, black spec, opening FEN, base seconds, increment seconds)``, optionally followed
#: by the log directory (``Path`` or ``None``: no per-move log) and the game index, which names
#: the log file ``<index>_<white>_vs_<black>.json``.
Job = Tuple[Any, ...]

#: Deepest iteration of a player when the tournament does not limit it (see make_player).
DEFAULT_MAX_DEPTH = 30

#: Thread-pool variables pinned to 1 in the workers, so that timed games do not compete for cores.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)


def paired_openings(n_pairs: int, plies: int = 6, seed: int = 0) -> List[str]:
    """Distinct, non-terminal positions reached by random legal play.

    :param n_pairs: Number of openings to return.
    :param plies: Random plies played from the initial position.
    :param seed: Seed of the random choice.
    :return: FENs of the openings.
    """
    rng = random.Random(seed)
    openings: List[str] = []
    seen = set()
    while len(openings) < n_pairs:
        board = chess.Board()
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over() or board.fen() in seen:
            continue
        seen.add(board.fen())
        openings.append(board.fen())
    return openings


def read_games(path: Path) -> List[Dict[str, Any]]:
    """Read the game records of a tournament file.

    A last line that does not parse, including one cut inside a multi-byte character, is ignored
    with a warning: it is the write of a hard kill. A broken line anywhere else means the file is
    corrupt and raises.

    :param path: JSON-lines file written by :func:`run_round_robin`.
    :return: One dict per game; an empty list if the file does not exist.
    :raises ValueError: If a line other than the last one is not valid JSON.
    """
    if not path.exists():
        return []
    lines = path.read_bytes().split(b"\n")
    last = max((number for number, line in enumerate(lines) if line.strip()), default=-1)
    records: List[Dict[str, Any]] = []
    for number, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            records.append(json.loads(raw.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if number == last:
                logger.warning("ignoring a truncated last line in %s", path)
                break
            raise ValueError(f"{path}: line {number + 1} is not valid JSON") from exc
    return records


def _repair_tail(path: Path) -> None:
    """Make the file end with a newline, so that the next append starts on a fresh line.

    A final line that parses but lacks its newline gets one and is kept. A partial line (cut by a
    hard kill, possibly inside a multi-byte character) is truncated away.

    :param path: JSON-lines file; left alone if it does not exist or ends with a newline.
    """
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    keep = data.rfind(b"\n") + 1
    try:
        json.loads(data[keep:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("dropping %d bytes of a truncated line from %s", len(data) - keep, path)
        with path.open("r+b") as handle:
            handle.truncate(keep)
    else:
        with path.open("ab") as handle:
            handle.write(b"\n")


def _check_time_control(
    records: Sequence[Dict[str, Any]], base: float, increment: float, path: Path
) -> None:
    """Refuse to mix games played at another time control into one file.

    :raises ValueError: If a record has a different base or increment (or none).
    """
    for record in records:
        if float(record.get("base", -1)) != float(base) or float(
            record.get("increment", -1)
        ) != float(increment):
            raise ValueError(
                f"{path} already holds games at another time control; "
                f"requested {base:g}+{increment:g} s. Use a different --out file."
            )


def check_specs(specs: Sequence[str]) -> None:
    """Build every player once in this process, so that a bad specification fails before any game.

    :param specs: Player specifications.
    :raises ValueError: If a player cannot be built (unknown name or missing saved model).
    """
    for spec in specs:
        try:
            make_player(spec, TimeManager())
        except Exception as exc:
            raise ValueError(f"cannot build player '{spec}': {exc}") from exc


@contextmanager
def single_threaded_env() -> Iterator[None]:
    """Set :data:`THREAD_VARIABLES` to 1 and restore the previous environment on exit.

    Workers spawned inside the block inherit the variables.
    """
    saved = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    os.environ.update({name: "1" for name in THREAD_VARIABLES})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class RecordingPlayer:
    """Wraps a player and records one entry per move (position, time spent, search statistics)."""

    def __init__(self, player: Any) -> None:
        self._player = player
        self.moves: List[Dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._player, name)

    def get_move(self, board: chess.Board, time_left: float) -> chess.Move:
        """Ask the inner player for a move and record it; its exceptions propagate."""
        fen = board.fen()
        started = time.perf_counter()
        move = self._player.get_move(board, time_left)
        seconds = time.perf_counter() - started
        stats = getattr(self._player, "last_search_stats", None)
        self.moves.append(
            {
                "ply": board.ply(),
                "color": "white" if board.turn == chess.WHITE else "black",
                "fen": fen,
                "move": move.uci(),
                "time_left_before": time_left,
                "move_seconds": seconds,
                "budget": getattr(self._player, "turn_budget", None),
                "completed_depth": getattr(stats, "completed_depth", None),
                "nodes": getattr(stats, "nodes_visited", None),
                "max_depth_reached": getattr(stats, "max_depth_reached", None),
            }
        )
        return move


def log_path(log_dir: Path, index: int, white: str, black: str) -> Path:
    """File of the per-move log of a game, with filename-safe player names."""
    safe = lambda name: re.sub(r"[^A-Za-z0-9_.-]", "_", name)  # noqa: E731
    return log_dir / f"{index}_{safe(white)}_vs_{safe(black)}.json"


def _play_game(job: Job) -> Dict[str, Any]:
    """Play one game in a worker process; a crash is returned as an ``error`` record.

    :param job: See :data:`Job`.
    :return: The game record.
    """
    white, black, fen, base, increment = job[:5]
    log_dir: Optional[Path] = job[5] if len(job) > 5 else None
    index: int = job[6] if len(job) > 6 else 0
    max_depth: int = job[7] if len(job) > 7 else DEFAULT_MAX_DEPTH
    record: Dict[str, Any] = {
        "white": white,
        "black": black,
        "fen": fen,
        "base": base,
        "increment": increment,
        "max_depth": max_depth,
    }
    try:
        white_player = RecordingPlayer(make_player(white, TimeManager(), max_depth))
        black_player = RecordingPlayer(make_player(black, TimeManager(), max_depth))
        result = GameManager(BotRegistry()).play_game(
            white_player, black_player, base, increment, starting_position=fen, verbose=False
        )
    except Exception as exc:  # a crashed game must not stop the tournament
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record.update(
        winner=result.winner,
        result=result.result,
        termination=result.termination,
        total_moves=result.total_moves,
    )
    if log_dir is not None:
        path = log_path(log_dir, index, white, black)
        path.parent.mkdir(parents=True, exist_ok=True)
        moves = sorted(white_player.moves + black_player.moves, key=lambda m: m["ply"])
        path.write_text(json.dumps({**record, "moves": moves}))
    return record


def _error_record(job: Job, message: str) -> Dict[str, Any]:
    """Record of a game that produced no result."""
    white, black, fen, base, increment = job[:5]
    return {
        "white": white,
        "black": black,
        "fen": fen,
        "base": base,
        "increment": increment,
        "error": message,
    }


def _run_pool(
    pending: Deque[Job],
    workers: int,
    play: Callable[[Job], Dict[str, Any]],
    emit: Callable[[Dict[str, Any]], None],
    deaths: Dict[Job, int],
) -> bool:
    """Play jobs from ``pending`` in one pool until it is empty or the pool breaks.

    CPython fails every in-flight future of a broken pool, not only the game that killed the
    worker. So the in-flight games of a broken pool are requeued without a record, and ``deaths``
    counts the broken pools each unfinished game was in flight for. While any such suspect exists,
    one game runs at a time, so that a death is blamed on that game alone.

    :param deaths: Broken pools per unfinished game. An entry is removed when its game gets a
        record.
    :return: True if the pool broke; the caller rebuilds it.
    """
    context = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context)
    in_flight: Dict[Future, Job] = {}
    broken = False
    try:
        while in_flight or (pending and not broken):
            limit = 1 if deaths else workers
            while pending and not broken and len(in_flight) < limit:
                job = pending.popleft()
                try:
                    in_flight[pool.submit(play, job)] = job
                except BrokenProcessPool:
                    pending.appendleft(job)
                    broken = True
            if not in_flight:
                break
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                job = in_flight.pop(future)
                try:
                    record = future.result()
                except BrokenProcessPool:
                    broken = True
                    deaths[job] = deaths.get(job, 0) + 1
                    if deaths[job] < MAX_DEATHS_PER_GAME:
                        pending.appendleft(job)
                        continue
                    record = _error_record(
                        job, f"worker process died {deaths[job]} times while running this game"
                    )
                except Exception as exc:
                    record = _error_record(job, f"{type(exc).__name__}: {exc}")
                deaths.pop(job, None)
                emit(record)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=not broken, cancel_futures=broken)
    return broken


def run_round_robin(
    specs: Sequence[str],
    openings: Sequence[str],
    base: float,
    increment: float,
    workers: int,
    out: Path,
    play: Callable[[Job], Dict[str, Any]] = _play_game,
    log_dir: Optional[Path] = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Dict[str, int]:
    """Play every pair on every opening twice, colours swapped, in parallel.

    Games already recorded in ``out`` (same white, black and opening) are skipped. Each new game
    is appended to ``out`` as soon as it finishes; every record carries ``base`` and ``increment``.
    A file recorded at another time control is refused, since its scores would be mixed.

    A game whose players raise is recorded with an ``"error"`` field. If a worker process dies, the
    games in flight in that pool are requeued without a record, because CPython fails all of them.
    A game that was in flight for :data:`MAX_DEATHS_PER_GAME` broken pools is recorded as an error
    and not retried. While such a suspect is unfinished, games run one at a time, so that the blame
    is unambiguous. After :data:`MAX_POOL_REBUILDS` rebuilds in a row with games still pending, the
    run stops with a ``RuntimeError``; rerun it to resume. A game played to a result resets that
    count, so a long run only gives up after consecutive broken pools without progress; error
    records do not reset it, so a systemic failure (every worker dies) ends the run.
    Errored games are excluded from the scores.

    The workers are started with ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, ``MKL_NUM_THREADS``,
    ``TF_NUM_INTRAOP_THREADS`` and ``TF_NUM_INTEROP_THREADS`` set to ``"1"``, so that timed games
    do not compete for cores; the parent's environment is restored afterwards. Pass
    ``workers <= os.cpu_count() - 1`` (the script does) to leave a core for the parent.

    All player specifications are built once in this process before any game starts. A truncated
    last line of ``out`` (left by a hard kill) is dropped before appending.

    On an exception or ``KeyboardInterrupt`` the pool is shut down without waiting for the queue
    and the exception is re-raised; the records already written stay valid.

    :param specs: Player specifications (see :func:`make_player`).
    :param openings: Starting FENs.
    :param base: Starting clock of each player, in seconds.
    :param increment: Seconds added per move.
    :param workers: Worker processes.
    :param out: JSON-lines file to append the games to.
    :param play: Plays one job; :func:`_play_game` unless a test substitutes its own.
    :param log_dir: Directory for the per-move log of each game (one JSON file per game); ``None``
        writes no logs.
    :return: ``played`` (new games), ``skipped`` (already recorded), ``errors`` (new games that
        produced no result) and ``games`` (records in ``out`` after the run).
    :raises ValueError: If a player specification cannot be built, or ``out`` holds another
        time control.
    :raises RuntimeError: If worker processes died more than :data:`MAX_POOL_REBUILDS` times
        while games were still pending.
    """
    specs = list(specs)
    jobs = [
        job
        for a, b in itertools.combinations(specs, 2)
        for fen in openings
        for job in ((a, b, fen, base, increment), (b, a, fen, base, increment))
    ]
    jobs = [(*job, log_dir, index, max_depth) for index, job in enumerate(jobs)]
    existing = read_games(out)
    _check_time_control(existing, base, increment, out)
    done = {(r["white"], r["black"], r["fen"]) for r in existing}
    pending: Deque[Job] = deque(job for job in jobs if job[:3] not in done)
    played = len(pending)
    counts = {"errors": 0}
    deaths: Dict[Job, int] = {}
    rebuilds = 0

    def emit(record: Dict[str, Any]) -> None:
        nonlocal rebuilds
        if "error" not in record:  # a game played to a result is progress; a crash is not
            rebuilds = 0
        counts["errors"] += "error" in record
        handle.write(json.dumps(record) + "\n")
        handle.flush()

    with single_threaded_env():
        check_specs(specs)
        if pending:
            out.parent.mkdir(parents=True, exist_ok=True)
            _repair_tail(out)
            with out.open("a") as handle:
                while pending:
                    if _run_pool(pending, workers, play, emit, deaths) and pending:
                        rebuilds += 1
                        logger.warning("worker process died; rebuilding the pool (%d)", rebuilds)
                        if rebuilds > MAX_POOL_REBUILDS:
                            raise RuntimeError(
                                f"worker processes died {rebuilds} times; giving up with "
                                f"{len(pending)} games unplayed. Rerun to resume."
                            )
    return {
        "played": played,
        "skipped": len(jobs) - played,
        "errors": counts["errors"],
        "games": len(read_games(out)),
    }


def _elo_from_score(score: float) -> float:
    """Elo difference implied by a score, ``-400 * log10(1/s - 1)`` with ``s`` clipped."""
    s = min(max(score, CLIP), 1 - CLIP)
    return -400 * math.log10(1 / s - 1)


def _fit_elo(
    counts: Dict[Tuple[str, str], int],
    points: Dict[str, float],
    games: Dict[str, int],
    anchor: str,
) -> Dict[str, float]:
    """Bradley-Terry ratings from all pairs, fitted by the minorise-maximise iteration.

    Scores per player are clipped like the pairwise ones, so no rating diverges. The ratings are
    anchored at ``anchor``.

    :param counts: Number of games per ordered pair ``(player, opponent)``.
    :param points: Points (wins + half draws) per player.
    :param games: Games per player.
    :param anchor: Player whose rating is fixed at 0.
    :return: Elo per player with games; empty if the anchor has no games.
    """
    players = [p for p in games if games[p] > 0]
    if anchor not in players:
        return {}
    weight = {p: min(max(points[p] / games[p], CLIP), 1 - CLIP) * games[p] for p in players}
    gamma = {p: 1.0 for p in players}
    for _ in range(2000):
        updated = {}
        for p in players:
            denominator = sum(
                counts[(p, q)] / (gamma[p] + gamma[q])
                for q in players
                if q != p and counts.get((p, q))
            )
            updated[p] = weight[p] / denominator
        scale = updated[anchor]
        change = max(abs(updated[p] / scale - gamma[p]) for p in players)
        gamma = {p: updated[p] / scale for p in players}
        if change < 1e-10:
            break
    return {p: 400 * math.log10(gamma[p]) for p in players}


def summarize(results: Sequence[Dict[str, Any]], anchor: str = ANCHOR) -> Dict[str, Any]:
    """Scores, W-D-L, Elo and 95% intervals of a set of game records.

    Crashed games (records with ``"error"``) are left out. For each player the score is
    ``wins + draws / 2`` over games. ``elo_vs_anchor`` uses the player's score against ``anchor``
    alone; ``elo`` is the Bradley-Terry fit over all pairs. The 95% interval of the score is the
    normal approximation from the variance of the per-game points; ``elo_interval`` is the
    half-width of the Elo implied by that score interval (an approximation, not a fitted interval).

    Caveats: scores entering the Elo formulas are clipped to ``[0.01, 0.99]``. The intervals treat
    games as independent, but the two games of a paired opening are correlated, so they are
    slightly optimistic. Crashed games are not retried.

    :param results: Game records as written by :func:`run_round_robin`.
    :param anchor: Player whose Elo is 0.
    :return: ``players`` (per player: games, wins, draws, losses, score in percent, ``ci95`` in
        percent, ``elo_vs_anchor``, ``elo``, ``elo_interval``), ``head_to_head`` (``[wins, draws,
        losses]`` of the row player against the column player), ``errors`` (number of crashed
        games), ``terminations`` (games per way of ending, crashed games excluded) and
        ``forfeit_games`` (games ended by a forfeit; they still count as normal results).
    :raises ValueError: If ``anchor`` takes part in none of the records.
    """
    if not any(anchor in (r["white"], r["black"]) for r in results):
        raise ValueError(f"the anchor player '{anchor}' must take part in the tournament")
    players = sorted({p for r in results for p in (r["white"], r["black"])})
    per_game: Dict[str, List[float]] = {p: [] for p in players}
    head = {p: {q: [0, 0, 0] for q in players if q != p} for p in players}
    counts: Dict[Tuple[str, str], int] = {}
    errors = 0
    terminations: Dict[str, int] = {}

    for record in results:
        if "error" in record:
            errors += 1
            continue
        if "termination" in record:
            terminations[record["termination"]] = terminations.get(record["termination"], 0) + 1
        white, black = record["white"], record["black"]
        white_points = {"white": 1.0, "black": 0.0, None: 0.5}[record["winner"]]
        per_game[white].append(white_points)
        per_game[black].append(1.0 - white_points)
        counts[(white, black)] = counts.get((white, black), 0) + 1
        counts[(black, white)] = counts.get((black, white), 0) + 1
        for player, opponent, points in (
            (white, black, white_points),
            (black, white, 1.0 - white_points),
        ):
            slot = 0 if points == 1.0 else 1 if points == 0.5 else 2
            head[player][opponent][slot] += 1

    games = {p: len(per_game[p]) for p in players}
    points = {p: sum(per_game[p]) for p in players}
    fitted = _fit_elo(counts, points, games, anchor)

    stats: Dict[str, Dict[str, Any]] = {}
    for p in players:
        n = games[p]
        entry: Dict[str, Any] = {
            "games": n,
            "wins": sum(record[0] for record in head[p].values()),
            "draws": sum(record[1] for record in head[p].values()),
            "losses": sum(record[2] for record in head[p].values()),
            "score": None,
            "ci95": None,
            "elo_vs_anchor": None,
            "elo": fitted.get(p),
            "elo_interval": None,
        }
        if n:
            mean = points[p] / n
            variance = statistics.fmean(x * x for x in per_game[p]) - mean * mean
            margin = 1.96 * math.sqrt(max(variance, 0.0) / n)
            entry["score"] = 100 * mean
            low, high = max(0.0, mean - margin), min(1.0, mean + margin)
            entry["ci95"] = [100 * low, 100 * high]
            entry["elo_interval"] = (_elo_from_score(high) - _elo_from_score(low)) / 2
            if p == anchor:
                entry["elo_vs_anchor"] = 0.0
            elif counts.get((p, anchor)):
                wins, draws, losses = head[p][anchor]
                score = (wins + 0.5 * draws) / (wins + draws + losses)
                entry["elo_vs_anchor"] = _elo_from_score(score)
        stats[p] = entry

    return {
        "players": stats,
        "head_to_head": head,
        "errors": errors,
        "terminations": terminations,
        "forfeit_games": sum(n for t, n in terminations.items() if t in FORFEIT_TERMINATIONS),
    }


def termination_section(summary: Dict[str, Any]) -> List[str]:
    """Markdown lines of the "How the games ended" section of the report.

    :param summary: Output of :func:`summarize`.
    :return: Lines of the section, to be placed right after the cross table.
    """
    terminations = summary["terminations"]
    total = sum(terminations.values())
    lines = ["## How the games ended", ""]
    if total:
        lines += ["| Termination | Games | Share |", "|---|---|---|"]
        for kind, n in sorted(terminations.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| {kind} | {n} | {100 * n / total:.1f}% |")
    else:
        lines.append("No games.")
    lines += [
        "",
        "Games decided by a forfeit (time, player error or illegal move): "
        f"{summary['forfeit_games']} of {total}. These are counted as results in the scores above.",
        "",
    ]
    return lines


def control_label(base: float, increment: float, max_depth: int = DEFAULT_MAX_DEPTH) -> str:
    """Key of a tournament in the ``tournament`` section of a card.

    :param base: Starting clock, in seconds.
    :param increment: Seconds per move.
    :param max_depth: Depth limit; a limited match is labelled by its depth, not by its clock.
    :return: ``depthN`` for a depth-limited match, else ``base+increment`` (e.g. ``60+1``).
    """
    if max_depth != DEFAULT_MAX_DEPTH:
        return f"depth{max_depth}"
    return f"{base:g}+{increment:g}"


def record_cards(
    summary: Dict[str, Any],
    specs: Sequence[str],
    anchor: str,
    base: float,
    increment: float,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> List[str]:
    """Write the entry of this tournament in the ``tournament`` section of the cards.

    The section holds one entry per control (see :func:`control_label`); the other controls
    are kept. An old flat section (one tournament, with ``base``) is migrated first. A player
    whose card does not exist yet is skipped (cards are written by training and by
    ``comparison.evaluate``, not created here), and so is one that played no game.

    :param summary: Output of :func:`summarize`.
    :param specs: Players of the tournament.
    :param anchor: Elo anchor of the summary.
    :param base: Starting clock, in seconds.
    :param increment: Seconds per move.
    :param max_depth: Depth limit of every player.
    :return: The players whose card was updated.
    """
    label = control_label(base, increment, max_depth)
    recorded = []
    for spec in specs:
        entry = summary["players"].get(spec)
        if not entry or not entry["games"]:
            continue
        if not card_path(spec).exists():
            if spec not in HEURISTICS:
                logger.warning("no model card for %s: tournament section not written", spec)
            continue
        card = load_card(spec)
        section = card.get("tournament") or {}
        if "base" in section:  # old layout: a single flat tournament
            old = section
            section = {control_label(old["base"], old.get("increment", 0.0)): old}
        section[label] = {
            "games": entry["games"],
            "score": entry["score"],
            "elo": entry["elo"],
            "elo_interval": entry["elo_interval"],
            "anchor": anchor,
            "base": base,
            "increment": increment,
            "max_depth": max_depth,
        }
        card["tournament"] = section
        save_card(spec, card)
        recorded.append(spec)
    return recorded


def size_for_budget(hours: float, workers: int, n_players: int, seconds_per_game: float) -> int:
    """Openings per pair that fit a time budget.

    One opening costs ``2 * pairs`` games, where ``pairs`` is the number of unordered pairs.

    :param hours: Wall-clock budget.
    :param workers: Games played at the same time.
    :param n_players: Number of players.
    :param seconds_per_game: Worker-seconds per game (wall time times workers over games).
    :return: Number of openings (rounded down; may be 0).
    :raises ValueError: If there are fewer than two players.
    """
    pairs = n_players * (n_players - 1) // 2
    if pairs == 0:
        raise ValueError("a round-robin needs at least two players")
    games_available = hours * 3600.0 * workers / seconds_per_game
    return int(games_available // (2 * pairs))
