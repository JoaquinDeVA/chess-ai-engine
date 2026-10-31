"""Tests of the Python evaluation of the laboratory: scaling, clipping and the training roundtrip."""

import chess
import numpy as np
import pandas as pd
import pytest

from chess_engine.models import CP_SCALE, get_model_class
from chess_lab.models import fit_stream, new_model
from chess_lab.models.python_eval import PythonEvaluator, cp_to_expected_score
from chess_lab.models.representation import Representation
from tests.conftest import material_cp, random_positions


class TestScale:
    def test_expected_score_is_centred_bounded_and_monotonic(self):
        assert cp_to_expected_score(0) == 0.5
        assert cp_to_expected_score(CP_SCALE) == cp_to_expected_score(10 * CP_SCALE) == 0.99
        assert cp_to_expected_score(-CP_SCALE) == 0.01
        scores = [cp_to_expected_score(cp) for cp in range(-1500, 1501, 100)]
        assert scores == sorted(scores)

    def test_expected_score_is_antisymmetric_around_one_half(self):
        for cp in (10, 250, 800):
            assert cp_to_expected_score(cp) + cp_to_expected_score(-cp) == pytest.approx(1.0)

    def test_evaluations_never_reach_the_terminal_values(self):
        """Regression B10: the search compares evaluations with terminal values 0, 0.5 and 1."""
        assert 0 < cp_to_expected_score(-1e9) and cp_to_expected_score(1e9) < 1


class TestMaterialLinear:
    def test_fit_save_load_roundtrip_learns_material(self, tmp_path, sample_boards):
        boards = random_positions(400, seed=5)
        train = pd.DataFrame(
            {
                "fen": [b.fen() for b in boards],
                "cp": [material_cp(b) for b in boards],
                "mate": np.nan,
                "white_to_move": [b.turn == chess.WHITE for b in boards],
            }
        )
        model_class = get_model_class("linear_material")
        model = new_model(model_class, Representation())
        summary = fit_stream(
            model,
            [train],
            train.head(100),
            {"l2": 0.0, "lr": 0.05, "max_epochs": 40, "patience": 40},
        )
        assert summary["train_rows"] == 400

        loaded = model_class.load(model.save(tmp_path / "material.npz"))
        predictions = PythonEvaluator(loaded).predict_cp_batch(sample_boards)
        truth = np.clip([material_cp(b) for b in sample_boards], -CP_SCALE, CP_SCALE)
        assert np.corrcoef(predictions, truth)[0, 1] > 0.9

    def test_predictions_are_clipped_to_the_cp_scale(self):
        model = get_model_class("linear_material")(weights=np.full(5, 10.0))
        extreme = chess.Board("4k3/8/8/8/8/8/PPPPPPPP/QQQQKQQQ w - - 0 1")
        assert abs(PythonEvaluator(model).predict_cp(extreme)) <= CP_SCALE
