"""Tests of scripts/bench/record_iteration_times.py (selection of moves and arguments)."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def script():
    path = (
        Path(__file__).resolve().parent.parent / "scripts" / "bench" / "record_iteration_times.py"
    )
    spec = importlib.util.spec_from_file_location("record_iteration_times", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_moves_of_returns_the_moves_of_one_player_in_play_order(script):
    game = {
        "white": "a",
        "black": "b",
        "moves": [
            {"ply": 2, "color": "white"},
            {"ply": 1, "color": "black"},
            {"ply": 0, "color": "white"},
        ],
    }
    assert [m["ply"] for m in script.moves_of(game, "a")] == [0, 2]
    assert [m["ply"] for m in script.moves_of(game, "b")] == [1]
    assert script.moves_of(game, "c") == []


def test_arguments(script):
    args = script.parse_args(["logs", "--player", "linear_material"])
    assert (args.games, args.player, args.out.name) == (
        2,
        "linear_material",
        "iteration_times.jsonl",
    )
