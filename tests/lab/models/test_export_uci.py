"""Tests of src/chess_lab/models/export_uci.py with small made-up models."""

import hashlib
import json

import numpy as np
import pytest

from chess_engine import _core
from chess_lab.models import export_uci
from chess_lab.models.definitions import (
    NnThreatsW64BothModel,
    NnW64BothModel,
    NnW128Ob8H16BothModel,
    get_model_class,
)
from chess_lab.models.export_uci import export_payload
from tests.conftest import CTX_MODELS, random_net, random_positions


def payload_of(model):
    return export_payload(model, "some_net", "0" * 64)


def test_network_payload_is_row_major():
    weights = np.arange(6.0).reshape(3, 2)
    model = NnW64BothModel([(weights, np.zeros(2)), (np.ones((2, 1)), np.zeros(1))])
    payload = payload_of(model)
    first = payload["layers"][0]
    assert (first["fan_in"], first["fan_out"]) == (3, 2)
    assert first["weights"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert payload["features"] == model.feature_set


def test_a_head_is_written_as_a_third_layer():
    model_class = get_model_class("net_pst_w128_ob8h16_both")
    layers = [(np.zeros(shape), np.zeros(shape[1])) for shape in model_class.layer_shapes()]
    payload = payload_of(model_class(layers))
    assert [(layer["fan_in"], layer["fan_out"]) for layer in payload["layers"]] == [
        (768, 128),
        (256, 128),
        (16, 8),
    ]


def test_both_perspectives_are_written_in_the_payload():
    layers = [(np.zeros((768, 4)), np.zeros(4)), (np.zeros((8, 1)), np.zeros(1))]
    assert payload_of(NnW64BothModel(layers))["perspective"] == "both"


def test_the_families_of_a_ctx_model_are_written_in_the_payload():
    model_class = get_model_class("net_ctx_mpk_w128_both")
    layers = [
        (np.zeros((model_class.input_rows(), 128)), np.zeros(128)),
        (np.zeros((256, 1)), np.zeros(1)),
    ]
    payload = payload_of(model_class(layers))
    assert payload["features"] == "pst_ctx"
    assert payload["families"] == ["material", "pawn_files", "king_buckets"]
    assert "families" not in payload_of(
        NnW64BothModel([(np.zeros((768, 4)), np.zeros(4)), (np.zeros((8, 1)), np.zeros(1))])
    )


def test_the_payload_carries_the_schema_of_the_core_and_its_provenance():
    layers = [(np.zeros((768, 8)), np.zeros(8)), (np.zeros((16, 1)), np.zeros(1))]
    payload = export_payload(NnW64BothModel(layers), "some_net", "ab" * 32)
    assert payload["schema"] == _core.MODEL_SCHEMA == 1
    assert (payload["name"], payload["source_sha256"]) == ("some_net", "ab" * 32)


@pytest.mark.parametrize(
    "model_class", [NnW64BothModel, NnThreatsW64BothModel, NnW128Ob8H16BothModel, *CTX_MODELS]
)
def test_the_core_scores_the_exported_file_like_the_model(model_class):
    model = random_net(seed=3, model_class=model_class)
    text = json.dumps(payload_of(model), allow_nan=False)
    loaded = _core.Evaluator.from_json(text)
    original = model.evaluator()
    for board in random_positions(20, seed=5):
        for white in (True, False):
            assert loaded.evaluate(board.fen(), white) == original.evaluate(board.fen(), white)


def test_the_core_rejects_a_file_without_schema_or_with_a_newer_one():
    layers = [(np.zeros((768, 8)), np.zeros(8)), (np.zeros((16, 1)), np.zeros(1))]
    payload = payload_of(NnW64BothModel(layers))
    _core.Evaluator.from_json(json.dumps(payload))
    with pytest.raises(ValueError, match="no schema"):
        _core.Evaluator.from_json(json.dumps({k: v for k, v in payload.items() if k != "schema"}))
    with pytest.raises(ValueError, match="newer"):
        _core.Evaluator.from_json(json.dumps({**payload, "schema": _core.MODEL_SCHEMA + 1}))


def export_with(monkeypatch, tmp_path, model):
    """Run ``main`` on ``model``, saved as the file ``source.npz`` of ``tmp_path``."""
    source = tmp_path / "source.npz"
    source.write_bytes(b"the bytes of the npz")
    monkeypatch.setattr(export_uci, "load_model", lambda name: model)
    monkeypatch.setattr(export_uci, "artifact_path", lambda name: source)
    return export_uci.main(["--model", "some_net", "--out", str(tmp_path / "out")])


def test_main_writes_the_name_and_the_hash_of_the_npz(monkeypatch, tmp_path):
    export_with(monkeypatch, tmp_path, random_net(seed=1))
    written = json.loads((tmp_path / "out" / "some_net.json").read_text())
    assert written["name"] == "some_net"
    assert written["source_sha256"] == hashlib.sha256(b"the bytes of the npz").hexdigest()


def test_a_network_the_core_cannot_read_is_not_written(monkeypatch, tmp_path):
    layers = [(np.zeros((768, 4)), np.zeros(4)), (np.zeros((8, 1)), np.zeros(1))]  # width 4
    with pytest.raises(ValueError):
        export_with(monkeypatch, tmp_path, NnW64BothModel(layers))
    assert not (tmp_path / "out").exists()


def test_a_weight_that_is_not_finite_is_not_written(monkeypatch, tmp_path):
    model = random_net(seed=1)
    model.layers[0][0][0, 0] = np.nan
    with pytest.raises(ValueError):
        export_with(monkeypatch, tmp_path, model)
    assert not (tmp_path / "out").exists()
