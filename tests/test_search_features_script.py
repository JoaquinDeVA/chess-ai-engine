"""Tests of scripts/bench/compare_search_features.py (switches and statistics; no real search)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from chess_engine.search import game_algorithms


@pytest.fixture(scope="module")
def script():
    path = (
        Path(__file__).resolve().parent.parent / "scripts" / "bench" / "compare_search_features.py"
    )
    spec = importlib.util.spec_from_file_location("compare_search_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _player():
    return SimpleNamespace(order_moves=lambda board, moves: sorted(moves), quiescence_plies=4)


def test_switched_off_replaces_and_restores_the_search_helpers(script):
    original = game_algorithms.probe_transposition_table
    player = _player()
    original_order = player.order_moves
    with script.switched_off(("table", "killers", "ordering", "quiescence"), player):
        assert game_algorithms.probe_transposition_table({}, "k", 1, 0.0, 1.0) == (None, 0.0, 1.0)
        assert game_algorithms.reorder_by_killer_moves(["a", "b"], 2, {}) == ["a", "b"]
        assert player.order_moves(None, ["b", "a"]) == ["b", "a"]
        assert player.quiescence_plies == 0
    assert game_algorithms.probe_transposition_table is original
    assert player.order_moves is original_order
    assert player.quiescence_plies == 4


def test_switched_off_with_nothing_changes_nothing(script):
    player = _player()
    with script.switched_off((), player):
        assert player.quiescence_plies == 4


def test_switches_are_restored_after_an_error(script):
    player = _player()
    with pytest.raises(RuntimeError), script.switched_off(("quiescence", "ordering"), player):
        raise RuntimeError("boom")
    assert player.quiescence_plies == 4


def test_share_equal(script):
    assert script.share_equal(["a", "b", "c", "d"], ["a", "x", "c", "y"]) == 0.5
    with pytest.raises(ValueError, match="same, non-empty"):
        script.share_equal(["a"], [])


def test_variants_start_with_the_baseline_and_end_with_everything_off(script):
    names = list(script.VARIANTS)
    assert names[0] == "baseline" and script.VARIANTS["baseline"] == ()
    assert set(script.VARIANTS[names[-1]]) == {"table", "killers", "ordering", "quiescence"}


def test_defaults_and_variant_selection(script):
    args = script.parse_args([])
    assert (args.positions, args.depth, args.variants) == (60, 4, None)
    assert script.parse_args(["--variants", "no killer moves"]).variants == "no killer moves"
