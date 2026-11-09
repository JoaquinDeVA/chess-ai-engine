"""The UCI client: the limits it sends and the moves the engine binary answers with."""

import json

import chess
import pytest

from chess_engine.gui.settings import AISettings
from chess_engine.uci import ENGINE, UciBot, UciEngine

pytestmark = pytest.mark.skipif(not ENGINE.exists(), reason="the uci binary is not built")


@pytest.fixture
def bot():
    player = UciBot(AISettings(max_depth=2, move_time=0.2, opening_book=False, tablebase=False))
    yield player
    player.close()


def test_the_engine_answers_with_a_legal_move(bot):
    board = chess.Board()
    move = bot.get_move(board, None, 0.0)
    assert move in board.legal_moves


def test_the_engine_knows_the_game_so_far(bot):
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    move = bot.get_move(board, None, 0.0)
    assert move in board.legal_moves


def test_limits_fixed_time_or_clock(bot):
    assert bot._limits(None, 0.0) == "depth 2 movetime 200"
    clocks = {chess.WHITE: 60.0, chess.BLACK: 58.5}
    bot.settings = AISettings(move_time=None)
    assert bot._limits(clocks, 0.1) == "wtime 60000 btime 58500 winc 100 binc 100"
    assert bot._limits(None, 0.0) == "movetime 3000"


def test_the_engine_plays_the_book_it_is_given(tmp_path):
    book = tmp_path / "book.json"
    book.write_text(json.dumps({chess.STARTING_FEN: "d2d4"}))
    engine = UciEngine([], 1, book=book)
    try:
        assert engine.bestmove(chess.Board(), "movetime 200") == chess.Move.from_uci("d2d4")
    finally:
        engine.close()
