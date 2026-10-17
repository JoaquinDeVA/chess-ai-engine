"""Tests for the depth-accuracy profiler."""

import chess
import pandas as pd
import pytest

from chess_engine.bots.iterative_heuristic_bot import IterativeHeuristicBot
from chess_engine.bots.iterative_model_book_bot import IterativeModelBookBot
from chess_engine.comparison import tradeoff
from chess_engine.comparison.tradeoff import fixed_time_manager, make_player, profile
from chess_engine.evaluation.heuristics import h1
from chess_engine.models import cards


def test_fixed_time_manager_budget_is_exact():
    manager = fixed_time_manager(2.5)
    budget = manager.turn_budget(2.5)
    assert budget == 2.5
    assert manager.hard_limit(budget) == 2.5


def test_make_player_builds_model_and_heuristic_players(stub_model):
    manager = fixed_time_manager(0.2)
    model_player = make_player(stub_model, manager, max_depth=4)
    assert isinstance(model_player, IterativeModelBookBot)
    assert model_player.max_depth == 4
    assert len(model_player.opening_book) == 0
    assert model_player.time_manager is manager
    player = make_player("h1", manager)
    assert isinstance(player, IterativeHeuristicBot)
    assert player.eval_function is h1
    assert player.time_manager is manager


def test_profile_reports_depth_and_agreement(stub_model):
    fens = [chess.Board().fen(), chess.Board("4k3/8/8/8/8/8/4P3/4K2R w K - 0 1").fen()]
    positions = pd.DataFrame({"fen": fens, "best_move": ["e2e4", None], "quiet": [True, False]})
    result = profile(stub_model, positions, 0.2)
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
    from chess_engine.models import base as models_base
    from chess_engine.models import registry

    monkeypatch.setattr(cards, "CARDS_DIR", tmp_path / "cards")
    monkeypatch.setattr(models_base, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(registry, "ARTIFACTS_DIR", tmp_path)
    return tmp_path
