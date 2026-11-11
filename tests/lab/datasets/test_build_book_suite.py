"""Tests of src/chess_lab/datasets/build_book_suite.py with tiny books."""

import chess

from chess_lab.datasets.build_book_suite import suite
from chess_lab.datasets.build_opening_book import position_key


def keys(*lines):
    """Book holding every position along the lines (the move stored does not matter here)."""
    book = {}
    for line in lines:
        board = chess.Board()
        book[position_key(board)] = line[0]
        for uci in line:
            board.push_uci(uci)
            book[position_key(board)] = uci
    return book


def test_only_positions_held_by_every_book_at_the_given_ply():
    old = keys(["e2e4", "e7e5"], ["d2d4", "d7d5"])
    new = keys(["e2e4", "e7e5"], ["e2e4", "c7c5"])
    fens = suite([old, new], plies=2)
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert fens == [board.fen()]


def test_no_positions_if_a_book_lacks_the_start():
    assert suite([{}], plies=2) == []
