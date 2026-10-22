#!/usr/bin/env python3
"""Measure what the endgame tablebases add: converting won endgames against perfect defence.

python scripts/bench/compare_endgame_tablebase.py --positions 60 --seconds 0.5

Random won positions with three to five pieces are generated (the side to move wins, according to
the tables). The winner is played twice: by a search bot that probes the tables and by the same bot
without them (a fixed time per move); the defender always plays the tablebase move, so it defends
perfectly. A game counts as converted when the winner mates within ``--max-moves`` moves of its own.

Writes ``data/reports/ENDGAME_TABLEBASE.md``. Nothing is written to the model cards.
"""

import argparse
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import chess
import chess.syzygy

from chess_engine.bots.iterative_heuristic_bot import IterativeHeuristicBot
from chess_engine.comparison.tradeoff import fixed_time_manager
from chess_engine.knowledge.endgame_tablebase import DEFAULT_TABLEBASE_DIR, EndgameTablebase

#: Material of the generated positions, as White's extra pieces and Black's.
MATERIALS: Tuple[Tuple[str, str], ...] = (
    ("Q", ""),
    ("R", ""),
    ("P", ""),
    ("BB", ""),
    ("BN", ""),
    ("RP", ""),
    ("R", "B"),
    ("Q", "R"),
    ("P", "P"),
)


def random_won_position(rng: random.Random, tables: EndgameTablebase) -> chess.Board:
    """A legal position where White, to move, wins according to the tables.

    :param rng: Source of randomness.
    :param tables: Open tablebases.
    :return: The position.
    """
    while True:
        white_extra, black_extra = rng.choice(MATERIALS)
        board = chess.Board(None)
        symbols = [("K", chess.WHITE), ("K", chess.BLACK)]
        symbols += [(s, chess.WHITE) for s in white_extra] + [(s, chess.BLACK) for s in black_extra]
        for square, (symbol, color) in zip(
            rng.sample(range(64), len(symbols)), symbols, strict=True
        ):
            board.set_piece_at(
                square, chess.Piece(chess.Piece.from_symbol(symbol).piece_type, color)
            )
        board.turn = chess.WHITE
        if not board.is_valid() or board.is_game_over():
            continue
        try:
            if tables._tables.probe_wdl(board) == 2:
                return board
        except (KeyError, chess.syzygy.MissingTableError):
            continue


def convert(
    start: chess.Board,
    winner: IterativeHeuristicBot,
    tables: EndgameTablebase,
    max_moves: int,
) -> Optional[int]:
    """Play a won endgame; the defender always plays the tablebase move.

    :param start: Position with White to move.
    :param winner: Bot that plays White.
    :param tables: Tablebases the defender plays from.
    :param max_moves: White moves allowed.
    :return: Number of White moves to checkmate, or ``None`` if White did not mate in time.
    """
    board = start.copy()
    winner.on_game_start(chess.WHITE, "defender")
    for move_number in range(1, max_moves + 1):
        board.push(winner.get_move(board, 60.0))
        if board.is_checkmate():
            return move_number
        if board.is_game_over():
            return None
        reply = tables.best_move(board)
        board.push(reply if reply is not None else next(iter(board.legal_moves)))
        if board.is_game_over():
            return None
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=60)
    parser.add_argument("--seconds", type=float, default=0.5, help="time per move of the search")
    parser.add_argument("--max-moves", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLEBASE_DIR)
    parser.add_argument("--report", type=Path, default=Path("data/reports/ENDGAME_TABLEBASE.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison and write the report.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    tables = EndgameTablebase(args.tables)
    rng = random.Random(args.seed)
    starts = [random_won_position(rng, tables) for _ in range(args.positions)]
    variants = {
        "search with tablebases": lambda: IterativeHeuristicBot(
            time_manager=fixed_time_manager(args.seconds), tablebase=tables, opening_book=None
        ),
        "search only": lambda: IterativeHeuristicBot(
            time_manager=fixed_time_manager(args.seconds), tablebase=None, opening_book=None
        ),
    }
    lines = [
        "# Converting won endgames against a perfect defender",
        "",
        f"{len(starts)} random won positions with three to five pieces (seed {args.seed}); the defender "
        f"plays the tablebase move; the winner has {args.seconds:g} s per move and "
        f"{args.max_moves} moves to mate.",
        "",
        "| winner | converted | mean moves to mate |",
        "|---|---|---|",
    ]
    for name, build in variants.items():
        results: List[Optional[int]] = []
        bot = build()
        for start in starts:
            results.append(convert(start, bot, tables, args.max_moves))
        won = [r for r in results if r is not None]
        lines.append(
            f"| {name} | {len(won)} of {len(results)} | "
            f"{(sum(won) / len(won)) if won else float('nan'):.1f} |"
        )
        print(lines[-1], flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
