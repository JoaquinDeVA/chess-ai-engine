"""Tests of src/chess_lab/scripts/bench/depth_in_game_logs.py."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def script():
    path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "chess_lab"
        / "scripts"
        / "bench"
        / "depth_in_game_logs.py"
    )
    spec = importlib.util.spec_from_file_location("depth_in_game_logs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _move(color, depth, seconds):
    return {"color": color, "completed_depth": depth, "move_seconds": seconds}


def test_load_moves_groups_by_player_and_skips_unknown_depth(script, tmp_path):
    game = {
        "white": "a",
        "black": "b",
        "moves": [_move("white", 4, 3.0), _move("black", 2, 0.1), _move("white", None, 1.0)],
    }
    (tmp_path / "0_a_vs_b.json").write_text(json.dumps(game))
    moves = script.load_moves(tmp_path)
    assert {k: len(v) for k, v in moves.items()} == {"a": 1, "b": 1}


def test_summarize_shares_and_fast_stops(script):
    moves = [_move("w", 2, 0.1), _move("w", 2, 5.0), _move("w", 3, 1.0), _move("w", 6, 9.0)]
    s = script.summarize(moves)
    assert s["moves"] == 4 and s["median"] == 2.5
    assert s["below3"] == 0.5 and s["below4"] == 0.75
    assert s["below3_fast"] == 1


def test_summarize_needs_moves(script):
    with pytest.raises(ValueError, match="no moves"):
        script.summarize([])


def test_main_reports_an_empty_folder(script, tmp_path, capsys):
    assert script.main([str(tmp_path)]) == 1
    assert "no move logs" in capsys.readouterr().out
