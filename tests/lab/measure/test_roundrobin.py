"""Tests of src/chess_lab/measure/roundrobin.py: pure parts only (no fastchess is run)."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def script():
    return importlib.import_module("chess_lab.measure.roundrobin")


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
    job = {
        "name": "x",
        "uci": "bin/ab12/uci",
        "a": ["option.Threads=2"],
        "b": [],
        "elo0": 0,
        "elo1": 5,
    }
    cmd = script.command(job, 100, 4, Path("c.json"), Path("m.pgn"), resume=False)
    assert cmd[0] == "./fastchess" and "option.Threads=2" in cmd and "elo1=5" in cmd
    assert cmd.count("cmd=./bin/ab12/uci") == 2
    assert cmd[cmd.index("-rounds") + 1] == "100"
    assert "file=openings.epd" in cmd
    own = script.command({**job, "openings": "book_suite.epd"}, 100, 4, Path("c"), Path("p"), False)
    assert "file=book_suite.epd" in own
    two = script.command(
        {**job, "uci_b": "bin/cd34/uci"}, 100, 4, Path("c.json"), Path("m.pgn"), False
    )
    assert two[two.index("name=A") - 1] == "cmd=./bin/ab12/uci"
    assert two[two.index("name=B") - 1] == "cmd=./bin/cd34/uci"
    assert script.command(job, 100, 4, Path("c.json"), Path("m.pgn"), resume=True) == [
        "./fastchess",
        "-config",
        "file=c.json",
        "outname=c.json",
    ]
