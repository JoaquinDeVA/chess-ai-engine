#!/usr/bin/env python3
"""Build the opening book from the Lichess evaluations.

python src/chess_lab/scripts/data/build_opening_book.py --raw data/lichess/raw --out data/opening_book.json

The raw shards hold a Stockfish evaluation for each of 125 million positions; most opening
positions are among them. The book is a walk through that data: from the starting position the
script plays every legal move, looks up the evaluation of the position it leads to, keeps the best
moves for the side that moves (those within ``--margin`` centipawns of the best, at most ``--top``)
and goes on from there, for ``--plies`` half-moves. The book stores, for every position it reached,
the best move. Lines that already favour one side by more than ``--eval-limit`` centipawns are not
followed, and each level keeps at most ``--cap-per-ply`` positions (the most balanced ones), which
bounds the size of the book.

Writes ``{fen: uci}`` (the first four FEN fields) to ``--out``; this is the one opening book of the
project (``knowledge/opening_book.py`` reads it).
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import chess
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

#: Score of a forced mate, minus the moves it takes (a faster mate is worth more).
MATE_SCORE = 100_000.0

Lookup = Callable[[str], Optional[float]]


def position_key(board: chess.Board) -> str:
    """First four FEN fields: the key of the raw shards and of the book."""
    return " ".join(board.fen().split()[:4])


def white_scores(cp: np.ndarray, mate: np.ndarray) -> np.ndarray:
    """Evaluation from White's point of view, in centipawns, mates turned into large scores.

    :param cp: Centipawns of each row (NaN where the row is a mate).
    :param mate: Moves to mate, positive when White mates (NaN where the row is not a mate).
    :return: One score per row.
    """
    return np.where(np.isnan(mate), cp, np.sign(mate) * (MATE_SCORE - np.abs(mate))).astype(
        np.float32
    )


def load_index(
    raw_dir: Path, min_pieces: int, min_depth: int, max_shards: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Index the positions of the shards that can belong to an opening.

    :param raw_dir: Folder with the ``part-*.parquet`` shards.
    :param min_pieces: Positions with fewer pieces on the board are skipped.
    :param min_depth: Evaluations searched to a smaller depth are skipped.
    :param max_shards: Read only this many shards (default: all).
    :return: ``(hashes, scores)`` sorted by hash; ``hash`` is Python's string hash, valid only in
        this process.
    """
    hashes: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    for path in sorted(raw_dir.glob("part-*.parquet"))[:max_shards]:
        table = pq.read_table(path, columns=["fen", "cp", "mate", "depth"])
        placement = pc.list_element(pc.split_pattern(table["fen"], " "), 0)
        pieces = pc.count_substring_regex(placement, "[A-Za-z]")
        keep = pc.and_(
            pc.greater_equal(pieces, min_pieces),
            pc.greater_equal(pc.fill_null(table["depth"], 0), min_depth),
        )
        table = table.filter(keep)
        if table.num_rows == 0:
            continue
        cp = table["cp"].to_numpy(zero_copy_only=False).astype(np.float64)
        mate = table["mate"].to_numpy(zero_copy_only=False).astype(np.float64)
        hashes.append(np.fromiter((hash(f) for f in table["fen"].to_pylist()), np.int64))
        scores.append(white_scores(cp, mate))
    all_hashes, all_scores = np.concatenate(hashes), np.concatenate(scores)
    order = np.argsort(all_hashes, kind="stable")
    return all_hashes[order], all_scores[order]


def make_lookup(hashes: np.ndarray, scores: np.ndarray) -> Lookup:
    """Function that gives the White score of a position key, or ``None`` if it is not indexed."""

    def lookup(key: str) -> Optional[float]:
        h = hash(key)
        i = int(np.searchsorted(hashes, h))
        if i < len(hashes) and hashes[i] == h:
            return float(scores[i])
        return None

    return lookup


def build_book(
    lookup: Lookup,
    plies: int,
    margin: float,
    top: int,
    eval_limit: float,
    cap_per_ply: int,
    start: Optional[chess.Board] = None,
) -> Dict[str, str]:
    """Walk the evaluated positions from the start and keep the best move of each.

    :param lookup: White score of a position key (``None`` when the position is not evaluated).
    :param plies: Half-moves to follow.
    :param margin: Moves within this many centipawns of the best are followed too.
    :param top: At most this many moves are followed from a position.
    :param eval_limit: Positions evaluated beyond this many centipawns for either side are not followed.
    :param cap_per_ply: Positions kept at each level (the ones closest to equal).
    :param start: Starting position (default: the initial position).
    :return: ``{position key: best move in UCI}``.
    """
    root = start.copy() if start is not None else chess.Board()
    frontier: Dict[str, chess.Board] = {position_key(root): root}
    book: Dict[str, str] = {}
    for _ in range(plies):
        children: Dict[str, Tuple[float, chess.Board]] = {}
        for key, board in frontier.items():
            scored: List[Tuple[float, chess.Move, str, chess.Board, float]] = []
            for move in list(board.legal_moves):
                board.push(move)
                child_key = position_key(board)
                white = lookup(child_key)
                if white is not None:
                    mover = white if board.turn == chess.BLACK else -white  # who just moved
                    scored.append((mover, move, child_key, board.copy(stack=False), white))
                board.pop()
            if not scored:
                continue
            scored.sort(key=lambda item: -item[0])
            best = scored[0][0]
            if best < -eval_limit:
                continue  # every move loses ground: not an opening line
            book[key] = scored[0][1].uci()
            for score, _, child_key, child_board, white in scored[:top]:
                if score >= best - margin and abs(white) <= eval_limit:
                    children.setdefault(child_key, (abs(white), child_board))
        ranked = sorted(children.items(), key=lambda item: item[1][0])[:cap_per_ply]
        frontier = {key: board for key, (_, board) in ranked if key not in book}
        if not frontier:
            break
    return book


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/lichess/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/opening_book.json"))
    parser.add_argument("--plies", type=int, default=16, help="half-moves followed")
    parser.add_argument("--margin", type=float, default=30.0, help="centipawns within the best")
    parser.add_argument("--top", type=int, default=3, help="moves followed from a position")
    parser.add_argument("--eval-limit", type=float, default=150.0, help="centipawns")
    parser.add_argument("--cap-per-ply", type=int, default=3000)
    parser.add_argument("--min-pieces", type=int, default=26)
    parser.add_argument("--min-depth", type=int, default=18)
    parser.add_argument("--max-shards", type=int, help="read only the first shards (for tests)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Index the shards, build the book and write it.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    hashes, scores = load_index(args.raw, args.min_pieces, args.min_depth, args.max_shards)
    print(f"{len(hashes):,} indexed positions")
    book = build_book(
        make_lookup(hashes, scores),
        args.plies,
        args.margin,
        args.top,
        args.eval_limit,
        args.cap_per_ply,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(book, indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{len(book):,} positions in the book -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
