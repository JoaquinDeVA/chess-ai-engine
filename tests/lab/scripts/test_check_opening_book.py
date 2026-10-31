"""Tests of src/chess_lab/scripts/data/check_opening_book.py with a fake engine (no network)."""

import importlib.util

import chess
import pytest

from tests.paths import ROOT

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


@pytest.fixture(scope="module")
def script():
    path = ROOT / "src" / "chess_lab" / "scripts" / "data" / "check_opening_book.py"
    spec = importlib.util.spec_from_file_location("check_opening_book", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def engine(pvs):
    return lambda fen, lines: {"pvs": pvs}


def test_a_move_among_the_best_lines_loses_the_difference(script):
    fetch = engine([{"moves": "e2e4 e7e5", "cp": 30}, {"moves": "d2d4 d7d5", "cp": 20}])
    assert script.judge_entry(START, "d2d4", fetch) == ("ok", 10.0)


def test_a_move_outside_the_lines_is_judged_by_its_child_position(script):
    def fetch(fen, lines):
        if fen.startswith("rnbqkbnr/pppppppp/8/8/8/7P"):  # after h2h3, Black to move
            return {"pvs": [{"moves": "e7e5", "cp": 10}]}  # Black +10, so White -10
        return {"pvs": [{"moves": "e2e4", "cp": 30}]}

    assert script.judge_entry(START, "h2h3", fetch) == ("ok", 40.0)


def test_an_illegal_move_and_an_unknown_position_are_reported_apart(script):
    assert script.judge_entry(START, "e2e5", engine([]))[0] == "illegal"
    assert script.judge_entry(START, "e2e4", lambda fen, lines: None)[0] == "missing"


def test_a_mate_counts_as_a_huge_score(script):
    assert script.line_score({"mate": 3}) > 90_000
    assert script.line_score({"mate": -3}) < -90_000


def test_sampling_is_reproducible_and_check_book_collects_failures(script):
    book = {}
    for first in list(chess.Board().legal_moves)[:10]:
        board = chess.Board()
        board.push(first)
        book[" ".join(board.fen().split()[:4])] = next(iter(board.legal_moves)).uci()
    assert script.sample_entries(book, 0.3, 1) == script.sample_entries(book, 0.3, 1)
    assert len(script.sample_entries(book, 0.3, 1)) == 3
    book_move = {"moves": "zzzz", "cp": 100}

    def fetch(fen, lines):
        played = next(iter(chess.Board(fen).legal_moves)).uci()
        return {"pvs": [book_move, {"moves": played, "cp": 0}]}

    result = script.check_book(book, 0.5, 1, 50.0, fetch)
    assert (result["sampled"], result["judged"], len(result["failed"])) == (5, 5, 5)
    assert result["mean_loss"] == 100.0
