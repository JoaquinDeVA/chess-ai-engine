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
import statistics
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterator, List, Sequence, Tuple

import chess

from chess_engine.bots.bot_registry import BotRegistry
from chess_engine.engine.game_manager import GameManager
from chess_engine.search.time_management import TimeManager
from chess_engine.training.tradeoff import make_player

logger = logging.getLogger(__name__)

#: Player whose rating is fixed at 0 Elo.
ANCHOR = "h3"

#: Scores are clipped to ``[CLIP, 1 - CLIP]`` before taking a logarithm.
CLIP = 0.01

#: Pools that may be rebuilt after a worker dies before the run gives up.
MAX_POOL_REBUILDS = 5

#: Broken pools a game may be in flight for before it is recorded as an error.
MAX_DEATHS_PER_GAME = 2

#: ``(white spec, black spec, opening FEN, base seconds, increment seconds)``.
Job = Tuple[str, str, str, float, float]

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


def _play_game(job: Job) -> Dict[str, Any]:
    """Play one game in a worker process; a crash is returned as an ``error`` record.

    :param job: See :data:`Job`.
    :return: The game record.
    """
    white, black, fen, base, increment = job
    record: Dict[str, Any] = {
        "white": white,
        "black": black,
        "fen": fen,
        "base": base,
        "increment": increment,
    }
    try:
        white_player = make_player(white, TimeManager())
        black_player = make_player(black, TimeManager())
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
    return record


def _error_record(job: Job, message: str) -> Dict[str, Any]:
    """Record of a game that produced no result."""
    white, black, fen, base, increment = job
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
) -> Dict[str, int]:
    """Play every pair on every opening twice, colours swapped, in parallel.

    Games already recorded in ``out`` (same white, black and opening) are skipped. Each new game
    is appended to ``out`` as soon as it finishes; every record carries ``base`` and ``increment``.
    A file recorded at another time control is refused, since its scores would be mixed.

    A game whose players raise is recorded with an ``"error"`` field. If a worker process dies, the
    games in flight in that pool are requeued without a record, because CPython fails all of them.
    A game that was in flight for :data:`MAX_DEATHS_PER_GAME` broken pools is recorded as an error
    and not retried. While such a suspect is unfinished, games run one at a time, so that the blame
    is unambiguous. After :data:`MAX_POOL_REBUILDS` rebuilds with games still pending, the run stops
    with a ``RuntimeError``; rerun it to resume. Errored games are excluded from the scores.

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
    existing = read_games(out)
    _check_time_control(existing, base, increment, out)
    done = {(r["white"], r["black"], r["fen"]) for r in existing}
    pending: Deque[Job] = deque(job for job in jobs if job[:3] not in done)
    played = len(pending)
    counts = {"errors": 0}
    deaths: Dict[Job, int] = {}

    def emit(record: Dict[str, Any]) -> None:
        counts["errors"] += "error" in record
        handle.write(json.dumps(record) + "\n")
        handle.flush()

    with single_threaded_env():
        check_specs(specs)
        if pending:
            out.parent.mkdir(parents=True, exist_ok=True)
            _repair_tail(out)
            rebuilds = 0
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
    counts: Dict[Tuple[str, str], int], points: Dict[str, float], games: Dict[str, int]
) -> Dict[str, float]:
    """Bradley-Terry ratings from all pairs, fitted by the minorise-maximise iteration.

    Scores per player are clipped like the pairwise ones, so no rating diverges. The ratings are
    anchored at :data:`ANCHOR`.

    :param counts: Number of games per ordered pair ``(player, opponent)``.
    :param points: Points (wins + half draws) per player.
    :param games: Games per player.
    :return: Elo per player with games; empty if the anchor has no games.
    """
    players = [p for p in games if games[p] > 0]
    if ANCHOR not in players:
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
        scale = updated[ANCHOR]
        change = max(abs(updated[p] / scale - gamma[p]) for p in players)
        gamma = {p: updated[p] / scale for p in players}
        if change < 1e-10:
            break
    return {p: 400 * math.log10(gamma[p]) for p in players}


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Scores, W-D-L, Elo and 95% intervals of a set of game records.

    Crashed games (records with ``"error"``) are left out. For each player the score is
    ``wins + draws / 2`` over games. ``elo_vs_h3`` uses the player's score against :data:`ANCHOR`
    alone; ``elo`` is the Bradley-Terry fit over all pairs. The 95% interval of the score is the
    normal approximation from the variance of the per-game points.

    Caveats: scores entering the Elo formulas are clipped to ``[0.01, 0.99]``. The intervals treat
    games as independent, but the two games of a paired opening are correlated, so they are
    slightly optimistic. Crashed games are not retried.

    :param results: Game records as written by :func:`run_round_robin`.
    :return: ``players`` (per player: games, wins, draws, losses, score in percent, ``ci95`` in
        percent, ``elo_vs_h3``, ``elo``), ``head_to_head`` (``[wins, draws, losses]`` of the row
        player against the column player) and ``errors`` (number of crashed games).
    :raises ValueError: If :data:`ANCHOR` takes part in none of the records.
    """
    if not any(ANCHOR in (r["white"], r["black"]) for r in results):
        raise ValueError(f"the anchor player '{ANCHOR}' must take part in the tournament")
    players = sorted({p for r in results for p in (r["white"], r["black"])})
    per_game: Dict[str, List[float]] = {p: [] for p in players}
    head = {p: {q: [0, 0, 0] for q in players if q != p} for p in players}
    counts: Dict[Tuple[str, str], int] = {}
    errors = 0

    for record in results:
        if "error" in record:
            errors += 1
            continue
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
    fitted = _fit_elo(counts, points, games)

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
            "elo_vs_h3": None,
            "elo": fitted.get(p),
        }
        if n:
            mean = points[p] / n
            variance = statistics.fmean(x * x for x in per_game[p]) - mean * mean
            margin = 1.96 * math.sqrt(max(variance, 0.0) / n)
            entry["score"] = 100 * mean
            entry["ci95"] = [100 * max(0.0, mean - margin), 100 * min(1.0, mean + margin)]
            if p == ANCHOR:
                entry["elo_vs_h3"] = 0.0
            elif counts.get((p, ANCHOR)):
                wins, draws, losses = head[p][ANCHOR]
                score = (wins + 0.5 * draws) / (wins + draws + losses)
                entry["elo_vs_h3"] = _elo_from_score(score)
        stats[p] = entry

    return {
        "players": stats,
        "head_to_head": head,
        "errors": errors,
    }


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
