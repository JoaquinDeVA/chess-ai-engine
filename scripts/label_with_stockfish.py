#!/usr/bin/env python3
"""Label positions with a local Stockfish and write the project's evaluations CSV.

Positions come from a text file (one FEN per line) or, with ``--random N``, from random play.

    python scripts/label_with_stockfish.py --random 20000 --depth 12 --output data/own_evals.csv
"""

import argparse
import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.lichess import write_csv  # noqa: E402
from training.stockfish import find_stockfish, label_positions  # noqa: E402


def random_fens(count: int, seed: int = 0, max_plies: int = 80):
    """Positions reached by random play.

    :param count: Number of positions.
    :param seed: Seed of the random generator.
    :param max_plies: Longest random game prefix.
    :return: Generator of FEN strings.
    """
    rng = random.Random(seed)
    produced = 0
    while produced < count:
        board = chess.Board()
        for _ in range(rng.randint(8, max_plies)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over():
            produced += 1
            yield board.fen()


def main() -> int:
    """Label the requested positions and write the CSV.

    :return: Process exit code.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fens", type=Path, help="text file with one FEN per line")
    source.add_argument("--random", type=int, metavar="N", help="label N random positions")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("data/own_evals.csv"))
    parser.add_argument("--engine", help="Stockfish executable (default: auto-detect)")
    args = parser.parse_args()

    engine = args.engine or find_stockfish()
    if engine is None:
        print(
            "Stockfish not found: install it (apt install stockfish) or pass --engine",
            file=sys.stderr,
        )
        return 1

    fens = (
        random_fens(args.random)
        if args.random
        else (line.strip() for line in args.fens.read_text().splitlines() if line.strip())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        written = write_csv(label_positions(fens, engine, args.depth), handle)
    print(f"Wrote {written:,} labelled positions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
