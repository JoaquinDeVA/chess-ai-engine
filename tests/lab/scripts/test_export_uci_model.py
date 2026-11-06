"""Tests of src/chess_lab/scripts/export_uci_model.py with small made-up models."""

import numpy as np
import pytest

from chess_lab.models.definitions import NnW32Model, PstTermsModel
from chess_lab.scripts.export_uci_model import export_payload


def test_linear_model_payload():
    model = PstTermsModel(np.arange(3.0), 0.5)
    payload = export_payload(model)
    assert payload["weights"] == [0.0, 1.0, 2.0] and payload["bias"] == 0.5
    assert payload["features"] == model.feature_set


def test_network_payload_is_row_major():
    weights = np.arange(6.0).reshape(3, 2)
    model = NnW32Model([(weights, np.zeros(2)), (np.ones((2, 1)), np.zeros(1))])
    first = export_payload(model)["layers"][0]
    assert (first["fan_in"], first["fan_out"]) == (3, 2)
    assert first["weights"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_other_models_are_rejected():
    with pytest.raises(SystemExit):
        export_payload(object())
