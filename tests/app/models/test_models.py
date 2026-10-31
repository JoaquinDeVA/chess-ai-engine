"""Tests of the app's model layer: registry, saved files and the shipped artifacts."""

import numpy as np
import pytest

from chess_engine.models import (
    available_models,
    get_model_class,
    load_model,
)
from chess_engine.models.base import EvaluationModel
from chess_engine.models.native import model_evaluator
from chess_engine.models.registry import register_model

WHITE_UP_A_QUEEN = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_UP_A_QUEEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"


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
    def test_core_scores_white_point_of_view(self, name):
        evaluator = model_evaluator(load_model(name))
        assert evaluator.evaluate(WHITE_UP_A_QUEEN, True) > evaluator.evaluate(
            BLACK_UP_A_QUEEN, True
        )

    def test_a_saved_model_keeps_its_representation(self, tmp_path):
        model = get_model_class("linear_material")(weights=np.ones(5), bias=0.5, rep={"x": 1})
        loaded = get_model_class("linear_material").load(model.save(tmp_path / "m.npz"))
        assert loaded.rep == {"x": 1} and loaded.bias == 0.5
        assert loaded.perspective == "side_to_move"
        assert loaded.count_parameters() == 6  # 5 coefficients + intercept


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
