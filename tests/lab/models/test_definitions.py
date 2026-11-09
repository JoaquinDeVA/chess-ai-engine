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
    assert {"net_pst_w32", "net_pst_w64", "net_pst_w64x16", "net_hybrid_w64"} <= set(
        available_models()
    )


def test_unknown_model_lists_the_valid_names():
    with pytest.raises(KeyError, match="available:.*net_pst_w32"):
        get_model_class("nope")


def test_missing_artifact_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model("net_pst_w32", tmp_path / "missing.npz")


def test_a_saved_model_keeps_its_representation(tmp_path):
    cls = get_model_class("net_pst_w32")
    model = new_model(cls, Representation())
    model.rep = {"x": 1}
    loaded = cls.load(model.save(tmp_path / "m.npz"))
    assert loaded.rep == {"x": 1}
    for (w, b), (w2, b2) in zip(loaded.layers, model.layers, strict=True):
        assert np.array_equal(w, w2) and np.array_equal(b, b2)
    assert loaded.count_parameters() == 768 * 32 + 32 + 32 + 1


def test_a_file_of_another_feature_set_is_refused(tmp_path):
    path = new_model(get_model_class("net_hybrid_w32"), Representation()).save(tmp_path / "m.npz")
    with pytest.raises(ValueError, match="features"):
        get_model_class("net_pst_w32").load(path)


def test_tagged_spec_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ARTIFACTS_DIR", tmp_path)
    cls = get_model_class("net_pst_w32")
    plain = new_model(cls, Representation())
    plain.save(tmp_path / "net_pst_w32.npz")
    tagged_model = new_model(cls, Representation())
    tagged_model.layers[-1] = (tagged_model.layers[-1][0], np.full(1, 0.5))
    tagged_model.save(tmp_path / "net_pst_w32@x.npz")

    assert load_model("net_pst_w32@x").layers[-1][1][0] == 0.5
    assert load_model("net_pst_w32").layers[-1][1][0] == 0.0
    with pytest.raises(FileNotFoundError):
        load_model("net_pst_w32@missing")
    with pytest.raises(KeyError):
        load_model("no_such_model@x")
