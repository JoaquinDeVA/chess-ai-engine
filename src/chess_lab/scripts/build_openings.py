"""Build the opening suite for the match tools: balanced positions 8 to 10 half-moves in.

Each candidate is a random legal playout of 8 to 10 half-moves from the start position, drawn
with a fixed seed. A local Stockfish, at a fixed depth, keeps the ones it scores within
``--limit`` pawns of equality. The same arguments give the same file (``sha256`` is printed).

Usage: ``python src/chess_lab/scripts/build_openings.py --out data/openings/openings.epd``
"""

import argparse
import hashlib
import random
from pathlib import Path
from typing import List

import chess
import chess.engine

MIN_PLY, MAX_PLY = 8, 10


def playouts(pool: int, seed: int) -> List[str]:
    """Distinct random positions 8 to 10 half-moves from the start.

    :param pool: How many distinct positions to draw.
    :param seed: Seed of the random playouts.
    :return: FENs, in draw order.
    """
    rng = random.Random(seed)
    seen, fens = set(), []
    while len(fens) < pool:
        board = chess.Board()
        for _ in range(rng.randint(MIN_PLY, MAX_PLY)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        fen = board.fen()
        if fen not in seen and board.ply() >= MIN_PLY and not board.is_game_over():
            seen.add(fen)
            fens.append(fen)
    return fens


def balanced(
    fens: List[str], engine: chess.engine.SimpleEngine, depth: int, limit: float
) -> List[str]:
    """Keep the positions Stockfish scores within ``limit`` pawns of equality.

    :param fens: Candidate positions.
    :param engine: An open Stockfish process.
    :param depth: Fixed search depth.
    :param limit: Largest accepted absolute score, in pawns.
    :return: The positions that pass, in the same order.
    """
    kept = []
    for fen in fens:
        info = engine.analyse(chess.Board(fen), chess.engine.Limit(depth=depth))
        score = info["score"].white().score(mate_score=100_000) / 100
        if abs(score) < limit:
            kept.append(fen)
    return kept


def parse_args() -> argparse.Namespace:
    """Read the command line.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--limit", type=float, default=0.5, help="largest |score| in pawns")
    parser.add_argument("--pool", type=int, default=6000, help="playouts to analyse")
    parser.add_argument("--count", type=int, default=1000, help="openings to keep")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Write one FEN per line, the form fastchess reads as an openings file."""
    args = parse_args()
    fens = playouts(args.pool, args.seed)
    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as engine:
        engine.configure({"Threads": 1})
        passed = balanced(fens, engine, args.depth, args.limit)
    chosen = random.Random(args.seed).sample(passed, min(args.count, len(passed)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f"{fen}\n" for fen in chosen), encoding="utf-8")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"{len(fens)} playouts, {len(passed)} balanced, {len(chosen)} written, sha256 {digest}")


if __name__ == "__main__":
    main()
