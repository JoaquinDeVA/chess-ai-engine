"""Players built from a spec."""

from chess_engine.bots.search_bot import SearchBot
from chess_engine.bots.time_manager import TimeManager
from chess_lab.players import (
    HEURISTICS,
    fixed_time_manager,
    make_player,
    split_techniques,
    validate_specs,
)


def test_fixed_time_manager_budget_is_exact():
    manager = fixed_time_manager(2.5)
    budget = manager.turn_budget(2.5)
    assert budget == 2.5
    assert manager.hard_limit(budget) == 2.5


def test_make_player_builds_model_and_heuristic_players(material_model):
    manager = fixed_time_manager(0.2)
    model_player = make_player(material_model, manager, max_depth=4)
    assert isinstance(model_player, SearchBot)
    assert model_player.max_depth == 4
    assert model_player.opening_book is None and model_player.tablebase is None
    assert model_player.time_manager is manager
    player = make_player("h1", manager)
    assert isinstance(player, SearchBot) and player.evaluation == "h1"
    assert player.time_manager is manager


def test_validate_specs_accepts_h1():
    assert "h1" in HEURISTICS
    validate_specs(["h1"])
    player = make_player("h1", fixed_time_manager(0.2))
    assert isinstance(player, SearchBot)


def test_specs_carry_the_techniques_of_a_player():
    assert split_techniques("linear_pst") == ("linear_pst", ())
    assert split_techniques("linear_pst+pvs,lmr") == ("linear_pst", ("pvs", "lmr"))
    assert len(split_techniques("h1+all")[1]) == 7


def test_make_player_switches_the_techniques_on_and_keeps_the_label():
    player = make_player("linear_material+pvs,lmr", TimeManager())
    assert player.techniques == ("pvs", "lmr") and player.name == "linear_material+pvs,lmr"
    assert make_player("linear_material", TimeManager()).techniques == ()
