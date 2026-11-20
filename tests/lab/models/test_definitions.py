"""Tests of the model definitions: the table of models, saved files and tagged specs."""

import numpy as np
import pytest

from chess_lab.models.definitions import (
    available_models,
    get_model_class,
    load_model,
)
from chess_lab.models.store import loader
from chess_lab.models.target import Representation
from chess_lab.training.neural import new_model


def test_every_model_is_in_the_table():
    assert set(available_models()) == {
        "net_pst_w64_both",
        "net_pst_w128_both",
        "net_pst_w256_both",
        "net_pst_threats_w64_both",
        "net_ctx_m_w128_both",
        "net_ctx_p_w128_both",
        "net_ctx_k_w128_both",
        "net_ctx_mpk_w128_both",
        "net_ctx_m_w256_both",
        "net_ctx_mpk_w256_both",
    }


@pytest.mark.parametrize("width", [128, 256])
def test_wider_models_share_the_inputs_and_double_the_output_layer(width):
    cls = get_model_class(f"net_pst_w{width}_both")
    assert cls.feature_set == get_model_class("net_pst_w64_both").feature_set
    assert cls.layer_shapes() == [(768, width), (2 * width, 1)]
    assert (
        new_model(cls, Representation()).count_parameters() == 768 * width + width + 2 * width + 1
    )


@pytest.mark.parametrize(
    ("letters", "families", "rows"),
    [
        ("m", ("material",), 768 + 30),
        ("p", ("pawn_files",), 768 + 32),
        ("k", ("king_buckets",), 4 * 768),
        ("mpk", ("material", "pawn_files", "king_buckets"), 4 * 768 + 30 + 32),
    ],
)
def test_ctx_models_read_the_families_they_name(letters, families, rows):
    cls = get_model_class(f"net_ctx_{letters}_w128_both")
    assert cls.feature_set == "pst_ctx"
    assert cls.families == families
    assert cls.layer_shapes() == [(rows, 128), (256, 1)]


def test_a_saved_ctx_model_remembers_its_families(tmp_path):
    cls = get_model_class("net_ctx_mpk_w128_both")
    path = new_model(cls, Representation()).save(tmp_path / "net_ctx_mpk_w128_both.npz")
    assert cls.load(path).families == cls.families
    with pytest.raises(ValueError, match="families"):
        get_model_class("net_ctx_k_w128_both").load(path)


def test_unknown_model_lists_the_valid_names():
    with pytest.raises(KeyError, match="available:.*net_pst_w64_both"):
        get_model_class("nope")


def test_missing_artifact_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model("net_pst_w64_both", tmp_path / "missing.npz")


def test_a_saved_model_keeps_its_representation(tmp_path):
    cls = get_model_class("net_pst_w64_both")
    model = new_model(cls, Representation())
    model.rep = {"x": 1}
    loaded = cls.load(model.save(tmp_path / "m.npz"))
    assert loaded.rep == {"x": 1}
    for (w, b), (w2, b2) in zip(loaded.layers, model.layers, strict=True):
        assert np.array_equal(w, w2) and np.array_equal(b, b2)
    assert loaded.count_parameters() == 768 * 64 + 64 + 128 + 1


def test_a_file_of_another_feature_set_is_refused(tmp_path):
    path = new_model(get_model_class("net_pst_threats_w64_both"), Representation()).save(
        tmp_path / "m.npz"
    )
    with pytest.raises(ValueError, match="features"):
        get_model_class("net_pst_w64_both").load(path)


def test_tagged_spec_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ARTIFACTS_DIR", tmp_path)
    cls = get_model_class("net_pst_w64_both")
    plain = new_model(cls, Representation())
    plain.save(tmp_path / "net_pst_w64_both.npz")
    tagged_model = new_model(cls, Representation())
    tagged_model.layers[-1] = (tagged_model.layers[-1][0], np.full(1, 0.5))
    tagged_model.save(tmp_path / "net_pst_w64_both@x.npz")

    assert load_model("net_pst_w64_both@x").layers[-1][1][0] == 0.5
    assert load_model("net_pst_w64_both").layers[-1][1][0] == 0.0
    with pytest.raises(FileNotFoundError):
        load_model("net_pst_w64_both@missing")
    with pytest.raises(KeyError):
        load_model("no_such_model@x")
