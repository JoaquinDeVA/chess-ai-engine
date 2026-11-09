"""Tests of src/chess_lab/scripts/export_uci_model.py with small made-up models."""

import numpy as np

from chess_lab.models.definitions import NnW32Model
from chess_lab.scripts.export_uci_model import export_payload


def test_network_payload_is_row_major():
    weights = np.arange(6.0).reshape(3, 2)
    model = NnW32Model([(weights, np.zeros(2)), (np.ones((2, 1)), np.zeros(1))])
    payload = export_payload(model)
    first = payload["layers"][0]
    assert (first["fan_in"], first["fan_out"]) == (3, 2)
    assert first["weights"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert payload["features"] == model.feature_set and "perspective" not in payload
