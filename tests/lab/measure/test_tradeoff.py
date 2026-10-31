"""Tests for the depth-accuracy profiler."""

import chess
import pandas as pd
import pytest

from chess_lab.measure import tradeoff
from chess_lab.measure.tradeoff import (
    profile,
)
from chess_lab.models import cards


def test_profile_reports_depth_and_agreement(material_model):
    fens = [chess.Board().fen(), chess.Board("4k3/8/8/8/8/8/4P3/4K2R w K - 0 1").fen()]
    positions = pd.DataFrame({"fen": fens, "best_move": ["e2e4", None], "quiet": [True, False]})
    result = profile(material_model, positions, 0.2)
    assert result["positions"] == 2
    for key in ("overall", "quiet", "non_quiet"):
        assert key in result
    overall = result["overall"]
    assert overall["mean_depth"] >= 1
    assert overall["median_depth"] >= 1
    assert overall["nodes_per_second"] > 0
    assert 0.0 <= overall["search_agreement"] <= 1.0
    assert result["quiet"]["positions"] == 1
    assert result["non_quiet"]["positions"] == 1


def _positions():
    return pd.DataFrame({"fen": [chess.STARTING_FEN], "best_move": ["e2e4"], "quiet": [True]})


def test_run_profiles_unknown_player_fails_before_profiling():
    with pytest.raises(ValueError, match="no_such_model"):
        tradeoff.run_profiles(["h1", "no_such_model"], [0.1], _positions())


@pytest.fixture
def cards_dir(tmp_path, monkeypatch):
    """Model cards in a temporary directory, with the artifacts of the tests in it too."""
    from chess_engine.models import loader

    monkeypatch.setattr(cards, "CARDS_DIR", tmp_path / "cards")
    monkeypatch.setattr(loader, "ARTIFACTS_DIR", tmp_path)
    return tmp_path
