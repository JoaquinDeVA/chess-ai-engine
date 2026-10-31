"""Every Rust feature set matches its Python extractor, and an unported one is rejected."""

import chess
import numpy as np
import pytest

from chess_engine import _core
from chess_lab.models.linear_features import FEATURE_SETS
from tests.paths import POSITIONS


def fens(count=300):
    import json

    rows = json.loads(POSITIONS.read_text())["positions"]
    return [entry["fen"] for entry in rows[:count]]


def test_every_rust_feature_set_has_a_python_version():
    assert set(_core.FEATURE_SETS) <= set(FEATURE_SETS)


@pytest.mark.parametrize("name", sorted(_core.FEATURE_SETS))
@pytest.mark.parametrize("perspective", ("white", "side_to_move"))
def test_rust_features_equal_python_features(name, perspective):
    features = FEATURE_SETS[name]
    try:
        size = features.size(perspective)
    except ValueError:
        pytest.skip(f"{name} does not define the {perspective} perspective")
    evaluator = _core.Evaluator.linear(name, perspective, np.zeros(size), 0.0)
    for fen in fens():
        board = chess.Board(fen)
        expected = features.active(board, perspective)
        got = evaluator.features(fen)
        assert list(got[0]) == list(expected[0]), fen
        assert got[1] == pytest.approx(expected[1], abs=1e-12), fen
