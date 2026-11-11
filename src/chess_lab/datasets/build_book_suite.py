#!/usr/bin/env python3
"""Build the opening suite of a book match: early positions that every book holds.

python -m chess_lab.datasets.build_book_suite --book data/knowledge/opening_book/opening_book.json \
    data/knowledge/opening_book/opening_book_deep.json --out data/openings/book_suite.epd

The usual suite (``build_openings``) holds random playouts that no book has, so an engine never
uses its book in those matches. This one walks every legal move from the start position for
``--plies`` half-moves, keeping the positions that all the given books hold, and writes those at
exactly ``--plies`` half-moves: each game starts inside both books and the engines follow them from
there. The positions are written in walk order (fastchess shuffles them with ``order=random``).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import chess

from chess_lab.datasets.build_opening_book import position_key


def suite(books: Sequence[Dict[str, str]], plies: int) -> List[str]:
    """Positions ``plies`` half-moves in, reached only through positions every book holds.

    :param books: ``{position key: uci}`` books.
    :param plies: Half-moves from the start position.
    :return: FENs (with move counters).
    """
    held = lambda board: all(position_key(board) in book for book in books)  # noqa: E731
    frontier = [chess.Board()] if held(chess.Board()) else []
    for _ in range(plies):
        seen, nxt = set(), []
        for board in frontier:
            for move in board.legal_moves:
                child = board.copy(stack=False)
                child.push(move)
                key = position_key(child)
                if key not in seen and held(child):
                    seen.add(key)
                    nxt.append(child)
        frontier = nxt
    return [board.fen() for board in frontier]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", type=Path, nargs="+", required=True, help="books (JSON)")
    parser.add_argument("--plies", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("data/openings/book_suite.epd"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the suite."""
    args = parse_args(argv)
    books = [
        {" ".join(fen.split()[:4]): uci for fen, uci in json.loads(p.read_text()).items()}
        for p in args.book
    ]
    fens = suite(books, args.plies)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(fen + "\n" for fen in fens), encoding="utf-8")
    print(f"{len(fens)} positions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
