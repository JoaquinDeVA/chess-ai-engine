"""Parallel round-robin tournament between players at equal time per move.

Every unordered pair of players meets on every opening twice, once with each colour. Games run
in a process pool and every worker builds its own players from the specification strings (bots
are not picklable). Each finished game is appended to a JSON-lines file, so an interrupted run
resumes where it stopped.
"""

import itertools
import json
import math
import multiprocessing
import os
import random
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import chess

from chess_engine.bots.bot_registry import BotRegistry
from chess_engine.engine.game_manager import GameManager
from chess_engine.search.time_management import TimeManager
from chess_engine.training.tradeoff import make_player

#: Player whose rating is fixed at 0 Elo.
ANCHOR = "h3"

#: Scores are clipped to ``[CLIP, 1 - CLIP]`` before taking a logarithm.
CLIP = 0.01

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

    :param path: JSON-lines file written by :func:`run_round_robin`.
    :return: One dict per game; an empty list if the file does not exist.
    """
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def _play_game(job: Tuple[str, str, str, float, float]) -> Dict[str, Any]:
    """Play one game in a worker process; a crash is returned as an ``error`` record.

    :param job: ``(white spec, black spec, opening FEN, base seconds, increment seconds)``.
    :return: The game record.
    """
    white, black, fen, base, increment = job
    record: Dict[str, Any] = {"white": white, "black": black, "fen": fen}
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


def run_round_robin(
    specs: Sequence[str],
    openings: Sequence[str],
    base: float,
    increment: float,
    workers: int,
    out: Path,
) -> Dict[str, int]:
    """Play every pair on every opening twice, colours swapped, in parallel.

    Games already recorded in ``out`` (same white, black and opening) are skipped. Each new game
    is appended to ``out`` as soon as it finishes. A game whose players crash is recorded with an
    ``"error"`` field and excluded from the scores.

    The workers are started with ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, ``MKL_NUM_THREADS``,
    ``TF_NUM_INTRAOP_THREADS`` and ``TF_NUM_INTEROP_THREADS`` set to ``"1"``, so that timed games
    do not compete for cores; the parent's environment is restored afterwards. Pass
    ``workers <= os.cpu_count() - 1`` (the script does) to leave a core for the parent.

    All player specifications are built once in this process before any game starts.

    :param specs: Player specifications (see :func:`make_player`).
    :param openings: Starting FENs.
    :param base: Starting clock of each player, in seconds.
    :param increment: Seconds added per move.
    :param workers: Worker processes.
    :param out: JSON-lines file to append the games to.
    :return: ``played`` (new games), ``skipped`` (already recorded), ``errors`` (new crashed
        games) and ``games`` (records in ``out`` after the run).
    :raises ValueError: If a player specification cannot be built.
    """
    specs = list(specs)
    jobs = [
        job
        for a, b in itertools.combinations(specs, 2)
        for fen in openings
        for job in ((a, b, fen, base, increment), (b, a, fen, base, increment))
    ]
    done = {(r["white"], r["black"], r["fen"]) for r in read_games(out)}
    pending = [job for job in jobs if job[:3] not in done]

    errors = 0
    with single_threaded_env():
        check_specs(specs)
        if pending:
            out.parent.mkdir(parents=True, exist_ok=True)
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(_play_game, job) for job in pending]
                with out.open("a") as handle:
                    for future in as_completed(futures):
                        record = future.result()
                        errors += "error" in record
                        handle.write(json.dumps(record) + "\n")
                        handle.flush()
    return {
        "played": len(pending),
        "skipped": len(jobs) - len(pending),
        "errors": errors,
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
