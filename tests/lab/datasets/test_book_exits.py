"""Tests of src/chess_lab/datasets/book_exits.py with tiny books and logs."""

import chess

from chess_lab.datasets import book_exits
from chess_lab.datasets.build_opening_book import position_key


def book_of(*lines):
    """Book holding every position along the given lines, with the line's next move."""
    book = {}
    for line in lines:
        board = chess.Board()
        for uci in line:
            book.setdefault(position_key(board), uci)
            board.push_uci(uci)
    return book


def test_read_games_keeps_only_the_longest_command_of_each_game():
    log = [
        "x << position startpos moves e2e4",
        "x << position startpos moves e2e4 e7e5 g1f3",
        "noise",
        "x << position startpos moves d2d4",
        "x << position startpos moves d2d4 d7d5 c2c4",
    ]
    assert book_exits.read_games(log) == [["e2e4", "e7e5", "g1f3"], ["d2d4", "d7d5", "c2c4"]]


def test_the_opponent_leaves_when_its_answer_is_out_of_the_book():
    book = book_of(["e2e4", "e7e5", "g1f3", "b8c6"])
    # Lista par: el motor juega con blancas.
    moves = ["e2e4", "c7c5", "g1f3", "d7d6"]
    assert book_exits.book_exit(book, moves) == (2, "opponent")


def test_the_line_ends_when_the_position_before_the_answer_is_not_in_the_book():
    book = book_of(["e2e4"])
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert book_exits.book_exit(book, moves) == (2, "line")


def test_summary_counts_reasons_and_median():
    book = book_of(["e2e4", "e7e5", "g1f3", "b8c6"])
    games = [["e2e4", "c7c5"], ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]]
    result = book_exits.summary(book, games)
    assert result["games"] == 2
    assert result["reasons"] == {"opponent": 1, "line": 1}
    assert result["median"] == 3
