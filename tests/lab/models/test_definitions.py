"""Tests of the model definitions: the table of models, saved files and tagged specs."""

import numpy as np
import pytest

from chess_lab.models import new_model
from chess_lab.models.definitions import (
    available_models,
    get_model_class,
    load_model,
)
from chess_lab.models.store import loader
from chess_lab.models.target import Representation


def test_every_model_is_in_the_table():
    assert {"linear_pst_terms", "net_pst_w32", "net_pst_w64", "net_pst_w64x16"} <= set(
        available_models()
    )


def test_unknown_model_lists_the_valid_names():
    with pytest.raises(KeyError, match="available:.*linear_pst_terms"):
        get_model_class("nope")


def test_missing_artifact_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model("linear_pst_terms", tmp_path / "missing.npz")


def test_a_saved_model_keeps_its_representation(tmp_path):
    model = get_model_class("linear_pst_terms")(weights=np.ones(790), bias=0.5, rep={"x": 1})
    loaded = get_model_class("linear_pst_terms").load(model.save(tmp_path / "m.npz"))
    assert loaded.rep == {"x": 1} and loaded.bias == 0.5
    assert loaded.perspective == "side_to_move"
    assert loaded.count_parameters() == 791  # 790 coefficients + intercept


def test_a_file_of_another_feature_set_is_refused(tmp_path):
    path = new_model(get_model_class("net_pst_w32"), Representation()).save(tmp_path / "m.npz")
    with pytest.raises(ValueError, match="features"):
        get_model_class("linear_pst_terms").load(path)


def test_tagged_spec_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ARTIFACTS_DIR", tmp_path)
    cls = get_model_class("linear_pst_terms")
    cls(weights=np.zeros(790), bias=0.0).save(tmp_path / "linear_pst_terms.npz")
    cls(weights=np.full(790, 0.5), bias=1.0).save(tmp_path / "linear_pst_terms@x.npz")

    tagged = load_model("linear_pst_terms@x")
    assert tagged.bias == 1.0 and tagged.weights[0] == 0.5
    assert load_model("linear_pst_terms").bias == 0.0
    with pytest.raises(FileNotFoundError):
        load_model("linear_pst_terms@missing")
    with pytest.raises(KeyError):
        load_model("no_such_model@x")
