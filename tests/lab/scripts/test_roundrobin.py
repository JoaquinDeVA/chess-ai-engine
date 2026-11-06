"""Tests of src/chess_lab/scripts/worker/roundrobin.py: pure parts only (no fastchess is run)."""

import importlib.util
from pathlib import Path

import pytest

from tests.paths import ROOT


@pytest.fixture(scope="module")
def script():
    path = ROOT / "src" / "chess_lab" / "scripts" / "worker" / "roundrobin.py"
    spec = importlib.util.spec_from_file_location("roundrobin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOG = """Elo: 1.23 +/- 4.50, nElo: 2.1 +/- 9.0
LOS: 70.00 %, DrawRatio: 40.00 %, PairsRatio: 1.02
Games: 400, Wins: 150, Losses: 140, Draws: 110, Points: 205.0 (51.25 %)
LLR: 0.53 (18.0%) (-2.94, 2.94) [0.00, 5.00]
"""


def test_parse_log_takes_the_last_block(script):
    text = "Games: 100,\nLLR: 0.10 (3%) (-2.94, 2.94) [0.00, 5.00]\n" + LOG
    out = script.parse_log(text)
    assert out["games"] == 400 and out["llr"] == 0.53 and out["elo"] == 1.23 and out["err"] == 4.5
    assert out["verdict"] is None


def test_parse_log_verdict_and_colours(script):
    text = "\x1b[32m" + LOG + "\x1b[0mSPRT ([0.00, 5.00]) completed - H1 was accepted\n"
    assert script.parse_log(text)["verdict"] == "H1"


def test_parse_log_empty(script):
    assert script.parse_log("")["llr"] is None


def test_rounds_played_and_next_rounds(script):
    config = {"stats": {"A vs B": {"wins": 10, "losses": 6, "draws": 4}}}
    assert script.rounds_played(config) == 10
    assert script.rounds_played({}) == 0
    assert script.next_rounds(10, 100, 6000) == 110
    assert script.next_rounds(5950, 100, 6000) == 6000


def test_command_first_turn_and_resume(script):
    job = {"name": "x", "a": ["option.Techniques=see"], "b": [], "elo0": 0, "elo1": 5}
    cmd = script.command(job, 100, 4, Path("c.json"), Path("m.pgn"), resume=False)
    assert cmd[0] == "./fastchess" and "option.Techniques=see" in cmd and "elo1=5" in cmd
    assert cmd[cmd.index("-rounds") + 1] == "100"
    assert script.command(job, 100, 4, Path("c.json"), Path("m.pgn"), resume=True) == [
        "./fastchess",
        "-config",
        "file=c.json",
        "outname=c.json",
    ]
