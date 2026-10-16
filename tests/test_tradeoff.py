"""Tests for the depth-accuracy profiler."""

import chess
import pandas as pd

from chess_engine.bots.iterative_heuristic_bot import IterativeHeuristicBot
from chess_engine.bots.iterative_model_book_bot import IterativeModelBookBot
from chess_engine.evaluation.heuristics import h2, h3
from chess_engine.training.tradeoff import fixed_time_manager, make_player, profile


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
    for spec, heuristic in (("h2", h2), ("h3", h3)):
        player = make_player(spec, manager)
        assert isinstance(player, IterativeHeuristicBot)
        assert player.eval_function is heuristic
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
