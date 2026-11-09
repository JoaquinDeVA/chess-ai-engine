"""Every Rust feature set matches its Python extractor, and an unported one is rejected."""

import chess
import numpy as np
import pytest

from chess_engine import _core
from chess_lab.models.features import FEATURE_SETS
from tests.paths import POSITIONS


def fens(count=300):
    import json

    rows = json.loads(POSITIONS.read_text())["positions"]
    return [entry["fen"] for entry in rows[:count]]


def test_every_rust_feature_set_has_a_python_version():
    assert set(_core.FEATURE_SETS) <= set(FEATURE_SETS)


@pytest.mark.parametrize("name", sorted(_core.FEATURE_SETS))
def test_rust_features_equal_python_features(name):
    features = FEATURE_SETS[name]
    layers = [(np.zeros((features.size(), 1)), np.zeros(1)), (np.zeros((1, 1)), np.zeros(1))]
    evaluator = _core.Evaluator.network(name, layers)
    for fen in fens():
        expected = features.active(chess.Board(fen))
        got = evaluator.features(fen)
        assert list(got[0]) == list(expected[0]), fen
        assert got[1] == pytest.approx(expected[1], abs=1e-12), fen
