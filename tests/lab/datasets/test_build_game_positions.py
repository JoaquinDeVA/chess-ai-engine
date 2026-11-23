"""Tests of the game-position set builder, with a small PGN and a fake engine."""

import chess
import chess.engine
import numpy as np

from chess_lab.datasets.build_game_positions import (
    game_positions,
    label_position,
    table_of,
    unique,
)

PGN = """[Event "a"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1

[Event "b"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *
"""


class FakeEngine:
    """Answers with a fixed score from the side to move's point of view and its first legal move."""

    def __init__(self, score):
        self.score = score

    def analyse(self, board, limit):
        move = next(iter(board.legal_moves))
        return {"score": chess.engine.PovScore(self.score, board.turn), "pv": [move]}


def test_game_positions_skip_the_finished_position_and_cap_each_game(tmp_path):
    pgn = tmp_path / "match.pgn"
    pgn.write_text(PGN)
    fens = game_positions(pgn, 10, np.random.default_rng(0))
    assert len(fens) == 3 + 4  # the mate after Qh4# is left out
    assert not any(chess.Board(fen).is_game_over() for fen in fens)
    assert len(game_positions(pgn, 2, np.random.default_rng(0))) == 4


def test_unique_ignores_the_move_counters():
    fen = chess.STARTING_FEN
    later = fen.replace(" 0 1", " 4 9")
    assert unique([fen, later, fen]) == [fen]


def test_label_position_is_from_whites_point_of_view():
    black_to_move = "4k3/8/8/8/8/8/8/4K2R b K - 0 1"
    fen, cp, mate, best = label_position(FakeEngine(chess.engine.Cp(50)), black_to_move, 1)
    assert fen == black_to_move and cp == -50.0 and mate != mate and best is not None
    _, cp, mate, _ = label_position(FakeEngine(chess.engine.Mate(2)), black_to_move, 1)
    assert cp != cp and mate == -2.0


def test_table_of_flags_a_capture_as_not_quiet():
    capture = "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"
    rows = [(capture, 10.0, float("nan"), "e4d5"), (capture, 10.0, float("nan"), "e1d1")]
    table = table_of(rows)
    assert list(table.columns) == ["fen", "cp", "mate", "best_move", "quiet"]
    assert table["quiet"].tolist() == [False, True]
