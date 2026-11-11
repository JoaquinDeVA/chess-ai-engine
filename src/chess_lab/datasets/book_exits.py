#!/usr/bin/env python3
"""Where and why the engine leaves the opening book, replaying the games of the Lichess bot.

python -m chess_lab.datasets.book_exits --book data/knowledge/opening_book/opening_book.json \
    --log data/results/lichess/lichess-bot.log*

The bot's debug log holds every ``position startpos moves ...`` command it sent to the engine; the
longest command of each game is the whole game. Each game is replayed with the book: the engine
follows it while its own positions are in it (as ``known_move`` in ``rust/src/bin/uci/main.rs``),
and the first position it lacks is the exit. The exit is the opponent's when the position before its
answer was in the book with another move (it played a move the book does not follow), and the
book's otherwise (the line ends: the book's own move, or a position it never held, led out). Several books can be compared on the same games.
"""

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import chess

from chess_lab.datasets.build_opening_book import position_key

_POSITION = re.compile(r"<< position startpos moves ?(.*)$")


def read_games(lines: Iterable[str]) -> List[List[str]]:
    """Games of the log: the move lists that the next command does not extend.

    :param lines: Lines of the bot's log, in order.
    :return: One list of UCI moves per game.
    """
    commands = [m.group(1).split() for line in lines if (m := _POSITION.search(line))]
    following = commands[1:] + [[]]
    return [
        moves for moves, nxt in zip(commands, following, strict=True) if nxt[: len(moves)] != moves
    ]


def book_exit(book: Dict[str, str], moves: Sequence[str]) -> Tuple[int, str]:
    """Half-move at which the engine leaves the book in a game, and why.

    The engine played the side to move after the last command (``moves`` has the opponent's last
    move, or none if the engine moved last and the game ended).

    :param book: ``{position key: uci}``.
    :param moves: UCI moves of the game.
    :return: ``(half-move, reason)``, reason ``"opponent"``, ``"line"`` or ``"end"`` (the game
        ended in the book).
    """
    board = chess.Board()
    engine = chess.WHITE if len(moves) % 2 == 0 else chess.BLACK
    for ply, uci in enumerate([*moves, None]):
        if board.turn == engine and position_key(board) not in book:
            if ply == 0:
                return 0, "line"
            before = board.copy()
            answer = before.pop().uci()
            left = book.get(position_key(before), answer) != answer
            return ply, "opponent" if left else "line"
        if uci is None:
            return ply, "end"
        board.push_uci(uci)
    raise AssertionError("unreachable")


def summary(book: Dict[str, str], games: Sequence[Sequence[str]]) -> Dict[str, object]:
    """Exit half-moves and reasons over the games.

    :param book: ``{position key: uci}``.
    :param games: Move lists.
    :return: ``{"games", "median", "reasons": Counter, "plies": Counter}``.
    """
    exits = [book_exit(book, moves) for moves in games]
    plies = [ply for ply, _ in exits]
    return {
        "games": len(exits),
        "median": statistics.median(plies) if plies else None,
        "reasons": Counter(reason for _, reason in exits),
        "plies": Counter(plies),
    }


def load_book(path: Path) -> Dict[str, str]:
    """The book with its keys reduced to the first four FEN fields."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {" ".join(fen.split()[:4]): uci for fen, uci in raw.items()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", type=Path, nargs="+", required=True, help="books (JSON)")
    parser.add_argument("--log", type=Path, nargs="+", required=True, help="logs of the bot")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the exits of each book over the games of the logs."""
    args = parse_args(argv)
    lines: List[str] = []
    for path in args.log:
        lines += path.read_text(encoding="utf-8", errors="replace").splitlines()
    games = read_games(lines)
    for path in args.book:
        result = summary(load_book(path), games)
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(result["reasons"].items()))
        plies = " ".join(f"{k}:{v}" for k, v in sorted(result["plies"].items()))
        print(f"{path}: {result['games']} games, median exit {result['median']}, {reasons}")
        print(f"  exits by half-move: {plies}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
