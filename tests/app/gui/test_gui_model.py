"""Tests of the Qt-free part of the GUI: the game on screen and the AI settings."""

import chess
import pytest

from chess_engine.gui.game_model import (
    Clock,
    GameModel,
    ai_accepts_draw,
)
from chess_engine.gui.settings import (
    DEFAULT_TECHNIQUES,
    AISettings,
    MatchSetup,
    PlayerSetup,
)


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_clock_runs_only_for_the_side_to_move_and_adds_the_increment():
    time = FakeTime()
    clock = Clock(60, 2, now=time)
    clock.start(chess.WHITE)
    time.now = 10
    assert clock.remaining(chess.WHITE) == 50
    assert clock.remaining(chess.BLACK) == 60
    clock.press(chess.WHITE)
    assert clock.remaining(chess.WHITE) == 52
    time.now = 25
    assert clock.remaining(chess.BLACK) == 45
    assert clock.flagged() is None
    time.now = 100
    assert clock.flagged() == chess.BLACK
    assert clock.remaining(chess.BLACK) == 0


def test_legal_moves_from_a_square_and_promotions():
    model = GameModel()
    targets = {move.to_square for move in model.legal_moves_from(chess.E2)}
    assert targets == {chess.E3, chess.E4}
    assert model.legal_moves_from(chess.E7) == []  # not the side to move
    promotion = GameModel("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    assert {m.promotion for m in promotion.legal_moves_from(chess.A7)} == {
        chess.QUEEN,
        chess.ROOK,
        chess.BISHOP,
        chess.KNIGHT,
    }


def test_make_move_records_san_and_rejects_illegal_moves():
    model = GameModel()
    assert model.make_move(chess.Move.from_uci("e2e4")) == "e4"
    assert model.make_move(chess.Move.from_uci("e7e5")) == "e5"
    assert model.san_moves == ["e4", "e5"]
    with pytest.raises(ValueError):
        model.make_move(chess.Move.from_uci("e2e4"))


def test_undo_restores_position_and_clocks():
    time = FakeTime()
    model = GameModel(clock=Clock(60, 0, now=time))
    time.now = 5
    model.make_move(chess.Move.from_uci("e2e4"))
    time.now = 12
    model.make_move(chess.Move.from_uci("e7e5"))
    assert model.undo(2) == 2
    assert model.board.fen() == chess.STARTING_FEN
    assert model.san_moves == []
    # the time spent thinking before the first move is not given back
    assert model.clock.remaining(chess.WHITE) == 55
    assert model.clock.remaining(chess.BLACK) == 60
    assert model.undo() == 0


def test_checkmate_ends_the_game_and_undo_reopens_it():
    model = GameModel()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        model.make_move(chess.Move.from_uci(uci))
    outcome = model.outcome()
    assert outcome.score == "0-1" and outcome.reason == "checkmate"
    assert outcome.winner == chess.BLACK
    with pytest.raises(ValueError):
        model.make_move(chess.Move.from_uci("a2a3"))
    model.undo()
    assert model.outcome() is None


def test_resignation_and_agreed_draw_are_final():
    model = GameModel()
    model.resign(chess.WHITE)
    assert model.outcome().score == "0-1"
    assert model.undo() == 0
    drawn = GameModel()
    drawn.agree_draw()
    assert drawn.outcome().score == "1/2-1/2"


def test_flag_loses_the_game():
    time = FakeTime()
    model = GameModel(clock=Clock(10, 0, now=time))
    time.now = 11
    assert model.outcome().score == "0-1" and model.outcome().reason == "timeout"


def test_flag_is_a_draw_when_the_opponent_cannot_mate():
    time = FakeTime()
    model = GameModel("k7/8/8/8/8/8/R7/K7 w - - 0 1", Clock(10, 0, now=time))
    model.clock.restore((10.0, 0.0), chess.BLACK)  # Black's time is over; White has a rook
    assert model.outcome().score == "1-0"
    other = GameModel("k7/8/8/8/8/8/p7/K7 w - - 0 1", Clock(10, 0, now=time))
    other.clock.restore((0.0, 10.0), chess.WHITE)  # White flags; Black has a pawn
    assert other.outcome().score == "0-1"
    bare = GameModel("k7/8/8/8/8/8/R7/K7 b - - 0 1", Clock(10, 0, now=time))
    bare.clock.restore((0.0, 10.0), chess.WHITE)  # White flags; Black has only a king
    assert bare.outcome().score == "1/2-1/2"


def test_ai_accepts_a_draw_only_when_not_ahead():
    board = chess.Board("k7/8/8/8/8/8/R7/K7 w - - 0 1")
    assert not ai_accepts_draw(board, chess.WHITE)
    assert ai_accepts_draw(board, chess.BLACK)


def test_pgn_has_the_names_and_the_result():
    model = GameModel()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        model.make_move(chess.Move.from_uci(uci))
    pgn = model.pgn("You", "Material logistic model")
    assert '[White "You"]' in pgn and '[Result "0-1"]' in pgn and "Qh4#" in pgn


def test_setup_names_and_defaults():
    assert AISettings().techniques == DEFAULT_TECHNIQUES and "lmr" not in AISettings().techniques
    setup = MatchSetup()
    assert setup.white.is_human and setup.has_humans() and setup.white.label == "You"
    assert not MatchSetup(PlayerSetup(AISettings()), PlayerSetup(AISettings())).has_humans()
