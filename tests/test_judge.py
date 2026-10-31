"""Tests of the judge: static accuracy and 1-ply best-move agreement."""

import chess
import numpy as np
import pandas as pd
import pytest

from chess_lab.comparison.judge import judge, one_ply_choice, static_metrics
from tests.conftest import MATE_IN_ONE, MaterialStubModel, material_cp, random_positions


class MaterialBatchModel(MaterialStubModel):
    """Material stub with a batched prediction, as every real model has."""

    def predict_cp_batch(self, boards):
        return np.array([material_cp(board) for board in boards], dtype=np.float64)


def test_spearman_perfect_and_reversed():
    truth = np.array([0.1, 0.4, 0.5, 0.9])
    assert static_metrics(truth * 2, truth)["spearman"] == pytest.approx(1.0)
    assert static_metrics(-truth, truth)["spearman"] == pytest.approx(-1.0)


def test_one_ply_takes_mate_in_one():
    board = chess.Board(MATE_IN_ONE)
    assert one_ply_choice(MaterialBatchModel(), board).uci() == "a1a8"


def test_one_ply_prefers_winning_the_queen():
    # White queen on d1 takes the hanging black queen on d4.
    board = chess.Board("4k3/8/8/8/3q4/8/8/3QK3 w - - 0 1")
    assert one_ply_choice(MaterialBatchModel(), board).uci() == "d1d4"


def test_judge_reports_slices():
    boards = random_positions(40, seed=7, max_plies=20)
    test = pd.DataFrame(
        {
            "fen": [b.fen() for b in boards],
            "cp": [float(material_cp(b)) for b in boards],
            "mate": [np.nan] * len(boards),
            "best_move": [next(iter(b.legal_moves)).uci() for b in boards],
            "quiet": [i % 3 != 0 for i in range(len(boards))],
            "white_to_move": [b.turn == chess.WHITE for b in boards],
        }
    )
    report = judge(MaterialBatchModel(), test, agreement_n=10)
    assert set(report) == {"all", "quiet", "non_quiet"}
    assert report["all"]["n"] == len(boards)
    assert report["quiet"]["n"] + report["non_quiet"]["n"] == report["all"]["n"]
    assert "spearman" in report["quiet"] and "best_move_agreement" in report["non_quiet"]
