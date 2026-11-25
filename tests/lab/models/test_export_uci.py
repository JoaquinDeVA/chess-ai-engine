"""Tests of src/chess_lab/models/export_uci.py with small made-up models."""

import numpy as np

from chess_lab.models.definitions import NnW64BothModel, get_model_class
from chess_lab.models.export_uci import export_payload


def test_network_payload_is_row_major():
    weights = np.arange(6.0).reshape(3, 2)
    model = NnW64BothModel([(weights, np.zeros(2)), (np.ones((2, 1)), np.zeros(1))])
    payload = export_payload(model)
    first = payload["layers"][0]
    assert (first["fan_in"], first["fan_out"]) == (3, 2)
    assert first["weights"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert payload["features"] == model.feature_set


def test_a_head_is_written_as_a_third_layer():
    model_class = get_model_class("net_pst_w128_ob8h16_both")
    layers = [(np.zeros(shape), np.zeros(shape[1])) for shape in model_class.layer_shapes()]
    payload = export_payload(model_class(layers))
    assert [(layer["fan_in"], layer["fan_out"]) for layer in payload["layers"]] == [
        (768, 128),
        (256, 128),
        (16, 8),
    ]


def test_both_perspectives_are_written_in_the_payload():
    layers = [(np.zeros((768, 4)), np.zeros(4)), (np.zeros((8, 1)), np.zeros(1))]
    assert export_payload(NnW64BothModel(layers))["perspective"] == "both"


def test_the_families_of_a_ctx_model_are_written_in_the_payload():
    model_class = get_model_class("net_ctx_mpk_w128_both")
    layers = [
        (np.zeros((model_class.input_rows(), 128)), np.zeros(128)),
        (np.zeros((256, 1)), np.zeros(1)),
    ]
    payload = export_payload(model_class(layers))
    assert payload["features"] == "pst_ctx"
    assert payload["families"] == ["material", "pawn_files", "king_buckets"]
    assert "families" not in export_payload(
        NnW64BothModel([(np.zeros((768, 4)), np.zeros(4)), (np.zeros((8, 1)), np.zeros(1))])
    )
