"""Tests of the saved models through the core: tagged specs and the shipped artifacts."""

import numpy as np
import pytest

from chess_lab.models.definitions import get_model_class, load_evaluator
from chess_lab.models.store import loader
from chess_lab.models.store.loader import installed_models, split_spec
from chess_lab.training.neural import new_model

WHITE_UP_A_QUEEN = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_UP_A_QUEEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"

# data/models/ is outside git: the tests of the trained files only run where they are installed.
needs_trained_models = pytest.mark.skipif(
    not (loader.ARTIFACTS_DIR / "net_pst_w32@e30.npz").exists(),
    reason="the trained models are not installed (data/models/ is outside git)",
)


def save_network(path, output_bias=0.0):
    """An untrained ``net_pst_w32`` file whose output bias is ``output_bias``."""
    model = new_model(get_model_class("net_pst_w32"))
    weights, _ = model.layers[-1]
    model.layers[-1] = (np.zeros_like(weights), np.full(1, output_bias))
    return model.save(path)


@needs_trained_models
def test_shipped_models_are_installed():
    assert {"net_hybrid_w64"} <= set(installed_models())


def test_missing_artifact_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_evaluator("net_pst_w32", tmp_path / "missing.npz")


@needs_trained_models
def test_core_scores_white_point_of_view():
    evaluator = load_evaluator("net_pst_w32@e30")
    assert evaluator.evaluate(WHITE_UP_A_QUEEN, True) > evaluator.evaluate(BLACK_UP_A_QUEEN, True)


def test_tagged_spec_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ARTIFACTS_DIR", tmp_path)
    save_network(tmp_path / "net_pst_w32.npz")
    save_network(tmp_path / "net_pst_w32@x.npz", output_bias=1.0)

    assert split_spec("net_pst_w32@x") == ("net_pst_w32", "x")
    assert split_spec("net_pst_w32") == ("net_pst_w32", None)
    with pytest.raises(ValueError):
        split_spec("net_pst_w32@")
    assert installed_models() == ["net_pst_w32"]
    assert load_evaluator("net_pst_w32@x").evaluate(WHITE_UP_A_QUEEN, True) > 0.5
    assert load_evaluator("net_pst_w32").evaluate(WHITE_UP_A_QUEEN, True) == 0.5
    with pytest.raises(FileNotFoundError):
        load_evaluator("net_pst_w32@missing")
