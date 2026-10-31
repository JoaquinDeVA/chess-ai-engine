"""Tests of the model definitions: the table of models, saved files and tagged specs."""

import numpy as np
import pytest

from chess_engine.models import loader
from chess_lab.models.definitions import (
    available_models,
    get_model_class,
    load_model,
)


def test_every_model_is_in_the_table():
    assert {"linear_pst", "linear_pst_terms", "linear_material", "linear_king_bucket"} <= set(
        available_models()
    )
    assert "net4_pst" in available_models()


def test_unknown_model_lists_the_valid_names():
    with pytest.raises(KeyError, match="available:.*linear_material"):
        get_model_class("nope")


def test_missing_artifact_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model("linear_material", tmp_path / "missing.npz")


def test_a_saved_model_keeps_its_representation(tmp_path):
    model = get_model_class("linear_material")(weights=np.ones(5), bias=0.5, rep={"x": 1})
    loaded = get_model_class("linear_material").load(model.save(tmp_path / "m.npz"))
    assert loaded.rep == {"x": 1} and loaded.bias == 0.5
    assert loaded.perspective == "side_to_move"
    assert loaded.count_parameters() == 6  # 5 coefficients + intercept


def test_a_file_of_another_feature_set_is_refused(tmp_path):
    path = get_model_class("linear_pst")(weights=np.zeros(768)).save(tmp_path / "m.npz")
    with pytest.raises(ValueError, match="features"):
        get_model_class("linear_pst_terms").load(path)


def test_tagged_spec_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ARTIFACTS_DIR", tmp_path)
    cls = get_model_class("linear_pst")
    cls(weights=np.zeros(768), bias=0.0).save(tmp_path / "linear_pst.npz")
    cls(weights=np.full(768, 0.5), bias=1.0).save(tmp_path / "linear_pst@x.npz")

    tagged = load_model("linear_pst@x")
    assert tagged.bias == 1.0 and tagged.weights[0] == 0.5
    assert load_model("linear_pst").bias == 0.0
    with pytest.raises(FileNotFoundError):
        load_model("linear_pst@missing")
    with pytest.raises(KeyError):
        load_model("no_such_model@x")
