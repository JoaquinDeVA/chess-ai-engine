"""Tests of src/chess_lab/scripts/bench/compare_endgame_tablebase.py with the 3-piece tables of tests/data."""

import importlib.util
import random

import chess
import pytest

from chess_engine.bots.search_bot import HEURISTIC, SearchBot
from chess_engine.knowledge.endgame_tablebase import EndgameTablebase
from tests.paths import ROOT, SYZYGY

TABLES = SYZYGY


@pytest.fixture(scope="module")
def script():
    path = ROOT / "src" / "chess_lab" / "scripts" / "bench" / "compare_endgame_tablebase.py"
    spec = importlib.util.spec_from_file_location("compare_endgame_tablebase", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tables():
    tb = EndgameTablebase(TABLES)
    yield tb
    tb.close()


def test_random_won_positions_are_legal_wins_for_white(script, tables, monkeypatch):
    monkeypatch.setattr(script, "MATERIALS", (("Q", ""), ("R", ""), ("P", "")))
    rng = random.Random(1)
    for _ in range(8):
        board = script.random_won_position(rng, tables)
        assert board.is_valid() and board.turn == chess.WHITE
        assert tables.wdl(board) == 2


def test_the_tablebase_winner_converts_kq_against_k(script, tables):
    start = chess.Board("8/8/8/8/8/2k5/8/K6Q w - - 0 1")
    winner = SearchBot(HEURISTIC, tablebase=tables, opening_book=None)
    assert script.convert(start, winner, tables, max_moves=30) is not None


def test_a_game_that_is_not_mated_in_time_returns_none(script, tables):
    start = chess.Board("8/8/8/8/8/2k5/8/K6Q w - - 0 1")
    assert (
        script.convert(
            start, SearchBot(HEURISTIC, tablebase=tables, opening_book=None), tables, max_moves=2
        )
        is None
    )
