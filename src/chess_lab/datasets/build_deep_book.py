#!/usr/bin/env python3
"""Build an opening book with a UCI engine, expanding the positions most likely to be reached first.

python -m chess_lab.datasets.build_deep_book --engine ~/stockfish/stockfish --budget 1000000 \
    --workers 8 --cache data/knowledge/opening_book/deep_cache.jsonl \
    --out data/knowledge/opening_book/opening_book_deep.json

The book is used with either colour, so every position has two reach probabilities: one for the
engine playing White and one for the engine playing Black. The engine plays its book move (the
best line of the analysis) with probability 1; the opponent spreads its probability over the lines
within ``--margin`` centipawns of the best one (at most ``--multipv``), with weights
``exp(-loss / --temperature)`` normalised to 1. The positions are analysed in order of the sum of
their two probabilities, until ``--budget`` positions are analysed, so forced lines go deep and
branching ones stay short.

Every analysis is appended to ``--cache`` (JSON lines); a run with the same arguments resumes from
it without asking the engine again. The book is ``{fen: uci}`` (first four FEN fields) with the
best move of every analysed position, the format of ``build_opening_book``.
"""

import argparse
import heapq
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import chess
import chess.engine

from chess_lab.datasets.build_opening_book import MATE_SCORE, position_key

#: Lines of an analysis: ``(uci, centipawns for the side to move)``, best first.
Lines = List[Tuple[str, float]]
Analyse = Callable[[chess.Board], Lines]


def reply_weights(lines: Lines, margin: float, temperature: float) -> List[Tuple[str, float]]:
    """Moves the opponent is expected to play, with their probabilities.

    :param lines: Analysis, best first.
    :param margin: Centipawns a line may lose against the best one.
    :param temperature: Centipawns that divide the probability by ``e``.
    :return: ``(uci, probability)`` summing to 1 (empty if there are no lines).
    """
    if not lines:
        return []
    best = lines[0][1]
    kept = [(uci, math.exp(-(best - cp) / temperature)) for uci, cp in lines if best - cp <= margin]
    total = sum(w for _, w in kept)
    return [(uci, w / total) for uci, w in kept]


def children(
    board: chess.Board, lines: Lines, reach: Tuple[float, float], margin: float, temperature: float
) -> List[Tuple[chess.Board, Tuple[float, float]]]:
    """Positions after an analysed one, with their reach probabilities.

    :param board: The analysed position.
    :param lines: Its analysis.
    :param reach: ``(engine White, engine Black)`` probabilities of reaching it.
    :param margin: See :func:`reply_weights`.
    :param temperature: See :func:`reply_weights`.
    :return: ``(position, (engine White, engine Black))`` for each followed move.
    """
    mover = 0 if board.turn == chess.WHITE else 1
    out = []
    for uci, weight in reply_weights(lines, margin, temperature):
        child = board.copy(stack=False)
        child.push_uci(uci)
        new = [0.0, 0.0]
        new[mover] = reach[mover] if uci == lines[0][0] else 0.0
        new[1 - mover] = reach[1 - mover] * weight
        out.append((child, (new[0], new[1])))
    return out


def build_book(
    analyse: Analyse,
    cache: Dict[str, Lines],
    budget: int,
    margin: float = 80.0,
    temperature: float = 30.0,
    workers: int = 1,
    on_analysed: Callable[[str, Lines], None] = lambda key, lines: None,
    log_every: int = 0,
) -> Dict[str, str]:
    """Analyse the most likely positions first and return the book.

    :param analyse: Analysis of a position (called from ``workers`` threads at once).
    :param cache: Analyses already done (``{position key: lines}``); new ones are added to it.
    :param budget: Positions analysed (cached ones included).
    :param margin: See :func:`reply_weights`.
    :param temperature: See :func:`reply_weights`.
    :param workers: Positions analysed at the same time.
    :param on_analysed: Called with each new analysis (to append it to the cache file).
    :param log_every: Print progress every this many positions (0: never).
    :return: ``{position key: best uci}``.
    """
    start = chess.Board()
    heap = [(-2.0, 0, start.fen(), (1.0, 1.0))]
    reach_of = {position_key(start): 2.0}
    done: Dict[str, str] = {}
    order, began, fresh = 1, time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while heap and len(done) < budget:
            batch = []
            while heap and len(batch) < min(workers * 4, budget - len(done)):
                _, _, fen, reach = heapq.heappop(heap)
                board = chess.Board(fen)
                key = position_key(board)
                if key in done or any(position_key(b) == key for b, _ in batch):
                    continue
                batch.append((board, reach))
            missing = [b for b, _ in batch if position_key(b) not in cache]
            for board, lines in zip(missing, pool.map(analyse, missing), strict=True):
                key = position_key(board)
                cache[key] = lines
                on_analysed(key, lines)
                fresh += 1
            for board, reach in batch:
                key = position_key(board)
                lines = cache[key]
                if not lines:
                    continue
                done[key] = lines[0][0]
                for child, child_reach in children(board, lines, reach, margin, temperature):
                    child_key, total = position_key(child), sum(child_reach)
                    if child_key in done or total <= reach_of.get(child_key, 0.0):
                        continue
                    reach_of[child_key] = total
                    heapq.heappush(heap, (-total, order, child.fen(), child_reach))
                    order += 1
                if log_every and len(done) % log_every == 0:
                    rate = fresh / max(time.time() - began, 1e-9)
                    print(
                        f"{len(done):,} positions ({fresh:,} analysed, {rate:.1f}/s), "
                        f"{len(heap):,} pending, next reach {-heap[0][0] if heap else 0:.2e}",
                        flush=True,
                    )
    return done


def engine_analyse(path: str, depth: int, multipv: int, hash_mb: int) -> Analyse:
    """Analysis by a UCI engine, one process per thread that calls it.

    :param path: Path of the engine.
    :param depth: Search depth.
    :param multipv: Lines asked for.
    :param hash_mb: Hash of each process, in MB.
    :return: The analysis function.
    """
    import threading

    local = threading.local()
    engines = []

    def analyse(board: chess.Board) -> Lines:
        if not hasattr(local, "engine"):
            local.engine = chess.engine.SimpleEngine.popen_uci(path)
            local.engine.configure({"Threads": 1, "Hash": hash_mb})
            engines.append(local.engine)
        infos = local.engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        return [
            (info["pv"][0].uci(), float(info["score"].pov(board.turn).score(mate_score=MATE_SCORE)))
            for info in infos
            if info.get("pv")
        ]

    analyse.engines = engines  # type: ignore[attr-defined]
    return analyse


def load_cache(path: Path) -> Dict[str, Lines]:
    """Analyses saved by earlier runs (a cut last line is ignored)."""
    cache: Dict[str, Lines] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[entry["fen"]] = [(uci, cp) for uci, cp in entry["lines"]]
    return cache


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="path of a UCI engine (Stockfish)")
    parser.add_argument("--budget", type=int, default=1_000_000, help="positions analysed")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--multipv", type=int, default=4, help="opponent moves considered")
    parser.add_argument("--margin", type=float, default=80.0, help="centipawns within the best")
    parser.add_argument("--temperature", type=float, default=30.0, help="centipawns")
    parser.add_argument("--workers", type=int, default=1, help="engine processes, one thread each")
    parser.add_argument("--hash", type=int, default=64, help="MB of hash per engine process")
    parser.add_argument(
        "--cache", type=Path, default=Path("data/knowledge/opening_book/deep_cache.jsonl")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/knowledge/opening_book/opening_book_deep.json")
    )
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the book and write it."""
    args = parse_args(argv)
    cache = load_cache(args.cache)
    print(f"{len(cache):,} analyses in {args.cache}", flush=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    analyse = engine_analyse(args.engine, args.depth, args.multipv, args.hash)
    with args.cache.open("a", encoding="utf-8") as sink:

        def save(key: str, lines: Lines) -> None:
            sink.write(json.dumps({"fen": key, "lines": lines}) + "\n")
            sink.flush()

        try:
            book = build_book(
                analyse,
                cache,
                args.budget,
                args.margin,
                args.temperature,
                args.workers,
                save,
                args.log_every,
            )
        finally:
            for engine in analyse.engines:  # type: ignore[attr-defined]
                engine.quit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(book, indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(book):,} positions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
