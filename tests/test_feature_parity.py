"""Every training feature family has a Rust version that matches it, or is declared slow."""

import logging
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_engine import _core
from chess_engine.models import linear_eval, linear_features
from chess_engine.models.native import model_evaluator

POSITIONS = Path(__file__).parent / "golden" / "positions.json"


def feature_classes():
    found, pending = [], list(linear_features.LinearFeatures.__subclasses__())
    while pending:
        cls = pending.pop()
        found.append(cls)
        pending.extend(cls.__subclasses__())
    return sorted(found, key=lambda cls: cls.name)


def fens(count=300):
    import json

    rows = json.loads(POSITIONS.read_text())["positions"]
    return [entry["fen"] for entry in rows[:count]]


@pytest.mark.parametrize("cls", feature_classes(), ids=lambda cls: cls.name)
@pytest.mark.parametrize("perspective", ("white", "side_to_move"))
def test_rust_features_equal_python_features(cls, perspective):
    if cls.name not in _core.FEATURE_SETS:
        pytest.skip(f"{cls.name} has no Rust version (slow path, see the next test)")
    features = cls()
    try:
        size = features.size(perspective)
    except ValueError:
        pytest.skip(f"{cls.name} does not define the {perspective} perspective")
    evaluator = _core.Evaluator.linear(cls.name, perspective, np.zeros(size), 0.0)
    for fen in fens():
        board = chess.Board(fen)
        expected = features.active(board, perspective)
        got = evaluator.features(fen)
        assert list(got[0]) == list(expected[0]), fen
        assert got[1] == pytest.approx(expected[1], abs=1e-12), fen


def test_a_family_without_rust_version_falls_back_with_a_warning(caplog):
    class Unported(linear_features.PstFeatures):
        name = "unported_test_features"

    class UnportedModel(linear_eval.LinearEvalModel):
        features = Unported()
        name = "unported_test"

    with caplog.at_level(logging.WARNING):
        evaluator = model_evaluator(UnportedModel())
    assert not evaluator.native
    assert "no Rust version" in caplog.text
