"""The UCI client: the limits it sends and the moves the engine binary answers with."""

import chess
import pytest

from chess_engine.gui.settings import AISettings
from chess_engine.uci import ENGINE, UciBot

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
