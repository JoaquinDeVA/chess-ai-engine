"""Tests of scripts/bench/compare_engine_versions.py (no games are played)."""

import importlib.util
import json
from pathlib import Path

import pytest

from chess_engine.comparison.tradeoff import make_player, split_techniques
from chess_engine.search.time_management import TimeManager


@pytest.fixture(scope="module")
def script():
    path = (
        Path(__file__).resolve().parent.parent / "scripts" / "bench" / "compare_engine_versions.py"
    )
    spec = importlib.util.spec_from_file_location("compare_engine_versions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_specs_carry_the_techniques_of_a_player():
    assert split_techniques("linear_pst") == ("linear_pst", ())
    assert split_techniques("linear_pst+pvs,lmr") == ("linear_pst", ("pvs", "lmr"))
    assert len(split_techniques("h1+all")[1]) == 6


def test_make_player_switches_the_techniques_on_and_keeps_the_label():
    player = make_player("linear_material+pvs,lmr", TimeManager())
    assert player.techniques == ("pvs", "lmr") and player.name == "linear_material+pvs,lmr"
    assert make_player("linear_material", TimeManager()).techniques == ()


def test_spec_of(script):
    assert script.spec_of("linear_pst", "") == "linear_pst"
    assert script.spec_of("linear_pst", "all") == "linear_pst+all"


def test_mean_depths_are_taken_from_the_move_logs_of_each_player(script, tmp_path):
    game = {
        "white": "a",
        "black": "b",
        "moves": [
            {"color": "white", "completed_depth": 6},
            {"color": "black", "completed_depth": 4},
            {"color": "white", "completed_depth": 8},
        ],
    }
    (tmp_path / "0_a_vs_b.json").write_text(json.dumps(game))
    assert script.mean_depths(tmp_path, ["a", "b"]) == {"a": 7.0, "b": 4.0}


def test_the_report_has_the_candidate_and_the_baseline(script):
    summary = {
        "players": {
            "m+all": {
                "games": 100,
                "wins": 55,
                "draws": 10,
                "losses": 35,
                "score": 60.0,
                "ci95": [50.2, 69.8],
                "elo_vs_anchor": 70.0,
            }
        }
    }
    lines = script.render("m", "m+all", summary, {"m": 6.0, "m+all": 7.5}, 10, 0.1)
    text = "\n".join(lines)
    assert "55-10-35" in text and "50.2 - 69.8" in text and "+70" in text and "7.50" in text


def test_defaults(script):
    args = script.parse_args([])
    assert (args.model, args.baseline, args.candidate, args.openings) == (
        "linear_pst_terms",
        "",
        "all",
        50,
    )
