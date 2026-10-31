"""Tests of the app's model loader: saved files, tagged specs and the shipped artifacts."""

import json

import numpy as np
import pytest

from chess_engine.models import (
    installed_models,
    load_evaluator,
    loader,
    split_spec,
)

WHITE_UP_A_QUEEN = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_UP_A_QUEEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"


def save_linear(path, features, size, bias=0.0, weight=0.0, rep=None):
    """A linear model file as the laboratory writes it."""
    meta = json.dumps({"features": features, "rep": rep or {}})
    np.savez(
        open(path, "wb"), weights=np.full(size, weight), bias=np.float64(bias), meta=np.array(meta)
    )


def test_shipped_models_are_installed():
    assert {"linear_material", "linear_pst_terms", "net4_pst"} <= set(installed_models())


def test_missing_artifact_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_evaluator("linear_material", tmp_path / "missing.npz")


@pytest.mark.parametrize("name", ["linear_material", "linear_pst_terms", "net4_pst"])
def test_core_scores_white_point_of_view(name):
    evaluator = load_evaluator(name)
    assert evaluator.evaluate(WHITE_UP_A_QUEEN, True) > evaluator.evaluate(BLACK_UP_A_QUEEN, True)


def test_tagged_spec_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ARTIFACTS_DIR", tmp_path)
    save_linear(tmp_path / "linear_pst.npz", "linear_pst", 768)
    save_linear(tmp_path / "linear_pst@x.npz", "linear_pst", 768, bias=1.0)

    assert split_spec("linear_pst@x") == ("linear_pst", "x")
    assert split_spec("linear_pst") == ("linear_pst", None)
    with pytest.raises(ValueError):
        split_spec("linear_pst@")
    assert installed_models() == ["linear_pst"]
    assert load_evaluator("linear_pst@x").evaluate(WHITE_UP_A_QUEEN, True) > 0.5
    assert load_evaluator("linear_pst").evaluate(WHITE_UP_A_QUEEN, True) == 0.5
    with pytest.raises(FileNotFoundError):
        load_evaluator("linear_pst@missing")


def test_a_feature_set_without_rust_version_is_rejected(tmp_path):
    save_linear(tmp_path / "unported.npz", "unported_test_features", 1)
    with pytest.raises(ValueError):
        load_evaluator("unported", tmp_path / "unported.npz")
