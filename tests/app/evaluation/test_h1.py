"""The classical reference ``h1``: symmetry, weights file and cache."""

import json
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_engine.evaluation import heuristics
from chess_engine.evaluation.chess_game_state import ChessGameState
from chess_engine.evaluation.heuristics import (
    FEATURE_NAMES,
    h1,
    load_h1_weights,
    reset_h1_cache,
)
from tests.conftest import random_positions

TUNED_WEIGHTS = {
    "material_diff_normalized": 0.6,
    "positional_diff_normalized": 0.1,
    "mobility_diff_normalized": 0.3,
    "center_diff_normalized": 0.05,
    "king_safety_diff_normalized": -0.2,
    "development_diff_normalized": 0.15,
}


@pytest.fixture(autouse=True)
def weights_file(tmp_path, monkeypatch):
    """Point ``h1`` at a weights file in a temporary directory, with a fresh cache."""
    path = tmp_path / "h1_weights.json"
    path.write_text(json.dumps({"weights": TUNED_WEIGHTS}))
    monkeypatch.setattr(heuristics, "H1_WEIGHTS_PATH", path)
    reset_h1_cache()
    yield path
    reset_h1_cache()


def states(count, seed):
    return [ChessGameState(board) for board in random_positions(count, seed=seed, max_plies=40)]


def test_h1_is_complementary_for_white_and_black():
    for state in states(50, seed=3):
        white = h1(state, chess.WHITE)
        black = h1(state, chess.BLACK)
        assert 0.01 <= white <= 0.99
        assert white + black == pytest.approx(1.0)


def test_missing_weights_file_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="src/chess_lab/scripts/data/fit_h1_weights.py"):
        load_h1_weights(tmp_path / "absent.json")


def test_weights_are_cached_until_reset(weights_file):
    assert load_h1_weights() == TUNED_WEIGHTS
    other = {**TUNED_WEIGHTS, "material_diff_normalized": 0.9}
    weights_file.write_text(json.dumps({"weights": other}))
    assert load_h1_weights() == TUNED_WEIGHTS  # served from the cache
    reset_h1_cache()
    assert load_h1_weights() == other


def test_shipped_weights_file_loads_with_the_feature_names():
    reset_h1_cache()
    weights = load_h1_weights(Path(heuristics.__file__).with_name("h1_weights.json"))
    assert set(weights) == set(FEATURE_NAMES)
    assert all(np.isfinite(value) for value in weights.values())
