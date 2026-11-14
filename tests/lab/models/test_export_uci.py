"""Tests of src/chess_lab/models/export_uci.py with small made-up models."""

import numpy as np

from chess_lab.models.definitions import NnW32Model, NnW64BothModel
from chess_lab.models.export_uci import export_payload


def test_network_payload_is_row_major():
    weights = np.arange(6.0).reshape(3, 2)
    model = NnW32Model([(weights, np.zeros(2)), (np.ones((2, 1)), np.zeros(1))])
    payload = export_payload(model)
    first = payload["layers"][0]
    assert (first["fan_in"], first["fan_out"]) == (3, 2)
    assert first["weights"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert payload["features"] == model.feature_set and "perspective" not in payload


def test_both_perspectives_are_written_in_the_payload():
    layers = [(np.zeros((768, 4)), np.zeros(4)), (np.zeros((8, 1)), np.zeros(1))]
    assert export_payload(NnW64BothModel(layers))["perspective"] == "both"
