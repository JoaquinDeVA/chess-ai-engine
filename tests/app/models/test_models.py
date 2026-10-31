"""Tests of the model layer: registry, scaling and the shipped artifacts."""

import chess
import numpy as np
import pandas as pd
import pytest

from chess_engine.models import (
    CP_SCALE,
    available_models,
    cp_to_expected_score,
    get_model_class,
    load_model,
)
from chess_engine.models.base import EvaluationModel
from chess_engine.models.registry import register_model
from tests.conftest import material_cp, random_positions

WHITE_UP_A_QUEEN = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_UP_A_QUEEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"


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


class TestRegistry:
    def test_builtin_models_are_registered(self):
        assert {
            "linear_material",
        } <= set(available_models())

    def test_unknown_model_lists_the_valid_names(self):
        with pytest.raises(KeyError, match="available:.*linear_material"):
            get_model_class("nope")

    def test_registering_a_name_twice_is_an_error(self):
        with pytest.raises(ValueError, match="already registered"):

            @register_model("linear_material")
            class Impostor(EvaluationModel):  # pragma: no cover - never instantiated
                pass

    def test_missing_artifact_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model("linear_material", tmp_path / "missing.npz")


class TestShippedModels:
    @pytest.mark.parametrize("name", ["linear_material", "linear_pst_terms", "net4_pst"])
    def test_sign_is_white_point_of_view(self, name):
        model = load_model(name)
        assert model.predict_cp(chess.Board(WHITE_UP_A_QUEEN)) > model.predict_cp(
            chess.Board(BLACK_UP_A_QUEEN)
        )

    def test_batch_prediction_matches_single_predictions(self, sample_boards):
        model = load_model("linear_pst_terms")
        batch = model.predict_cp_batch(sample_boards[:8])
        single = [model.predict_cp(board) for board in sample_boards[:8]]
        np.testing.assert_allclose(batch, single, atol=1e-2)


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
        model = get_model_class("linear_material")()
        summary = model.fit_stream(
            [train], train.head(100), {"l2": 0.0, "lr": 0.05, "max_epochs": 40, "patience": 40}
        )
        assert summary["train_rows"] == 400

        path = model.save(tmp_path / "material.npz")
        loaded = get_model_class("linear_material").load(path)
        predictions = loaded.predict_cp_batch(sample_boards)
        truth = np.clip([material_cp(b) for b in sample_boards], -CP_SCALE, CP_SCALE)
        assert np.corrcoef(predictions, truth)[0, 1] > 0.9
        assert loaded.count_parameters() == 6  # 5 coefficients + intercept

    def test_predictions_are_clipped_to_the_cp_scale(self):
        model = get_model_class("linear_material")(weights=np.full(5, 10.0))
        extreme = chess.Board("4k3/8/8/8/8/8/PPPPPPPP/QQQQKQQQ w - - 0 1")
        assert abs(model.predict_cp(extreme)) <= CP_SCALE


def test_tagged_spec_round_trip(tmp_path, monkeypatch):
    from chess_engine.models import base as models_base
    from chess_engine.models import registry

    monkeypatch.setattr(models_base, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(registry, "ARTIFACTS_DIR", tmp_path)
    cls = get_model_class("linear_pst")
    cls(weights=np.zeros(768), bias=0.0).save(tmp_path / "linear_pst.npz")
    cls(weights=np.full(768, 0.5), bias=1.0).save(tmp_path / "linear_pst@x.npz")

    assert registry.split_spec("linear_pst@x") == ("linear_pst", "x")
    assert registry.split_spec("linear_pst") == ("linear_pst", None)
    with pytest.raises(ValueError):
        registry.split_spec("linear_pst@")
    tagged = load_model("linear_pst@x")
    assert tagged.bias == 1.0 and tagged.weights[0] == 0.5
    plain = load_model("linear_pst")
    assert plain.bias == 0.0
    with pytest.raises(FileNotFoundError):
        load_model("linear_pst@missing")
    with pytest.raises(KeyError):
        load_model("no_such_model@x")
