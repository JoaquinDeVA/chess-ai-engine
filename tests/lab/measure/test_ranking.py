"""Tests of the move-ordering judge."""

import chess
import numpy as np
import pandas as pd
import pytest

from chess_lab.measure.ranking import judge_ranking, kendall_tau, move_scores
from tests.conftest import MATE_IN_ONE, MaterialStubModel

# White queen takes the hanging black queen, or a quiet king move, or a rook move that loses nothing.
QUEEN_FEN = "4k3/8/8/8/3q4/8/8/3QK3 w - - 0 1"


def test_kendall_tau_perfect_reversed_and_constant():
    truth = np.array([0.1, 0.4, 0.5, 0.9])
    assert kendall_tau(truth * 2, truth) == pytest.approx(1.0)
    assert kendall_tau(-truth, truth) == pytest.approx(-1.0)
    assert np.isnan(kendall_tau(truth, np.ones(4)))


def test_a_mating_move_scores_above_everything():
    board = chess.Board(MATE_IN_ONE)
    moves = list(board.legal_moves)
    scores = move_scores(MaterialStubModel(), board, moves)
    assert moves[int(np.argmax(scores))].uci() == "a1a8"


def _table(fen: str, cps: dict[str, float], quiet: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fen": fen,
            "move": list(cps),
            "cp": list(cps.values()),
            "mate": np.nan,
            "quiet": quiet,
        }
    )


def test_judge_ranking_agrees_when_the_truth_follows_material():
    board = chess.Board(QUEEN_FEN)
    # Stockfish-like scores: capturing the queen is far best, every other move is worse and equal.
    cps = {m.uci(): (900.0 if m.uci() == "d1d4" else 0.0) for m in board.legal_moves}
    report = judge_ranking(MaterialStubModel(), _table(QUEEN_FEN, cps))
    assert report["all"]["top1"] == 1.0
    assert report["all"]["top3"] == 1.0
    assert report["all"]["tau"] > 0.5
    assert report["all"]["positions"] == 1
    assert report["non_quiet"]["positions"] == 0


def test_judge_ranking_penalises_a_reversed_truth():
    board = chess.Board(QUEEN_FEN)
    cps = {m.uci(): (-900.0 if m.uci() == "d1d4" else 0.0) for m in board.legal_moves}
    report = judge_ranking(MaterialStubModel(), _table(QUEEN_FEN, cps, quiet=False))
    assert report["all"]["top1"] == 0.0
    assert report["all"]["tau"] < 0.0
    assert report["non_quiet"]["positions"] == 1
