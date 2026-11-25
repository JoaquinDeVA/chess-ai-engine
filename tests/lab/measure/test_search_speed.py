"""Tests of the search-speed timing, against a fake UCI engine."""

import stat
import sys

from chess_lab.measure import search_speed

FAKE_ENGINE = """#!{python}
import sys
log = open({log!r}, "a")
for line in sys.stdin:
    log.write(line)
    log.flush()
    if line.startswith("isready"):
        print("readyok", flush=True)
    elif line.startswith("go"):
        print("info depth 1", flush=True)
        print("bestmove e2e4", flush=True)
    elif line.startswith("quit"):
        break
"""


def test_seconds_to_depth_sends_the_options_and_one_search_per_position(tmp_path, monkeypatch):
    log = tmp_path / "received.txt"
    engine = tmp_path / "fake_uci"
    engine.write_text(FAKE_ENGINE.format(python=sys.executable, log=str(log)))
    engine.chmod(engine.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(search_speed, "BINARY", engine)

    fens = ["8/8/8/8/8/8/8/K6k w - - 0 1", "8/8/8/8/8/8/8/k6K b - - 0 1"]
    seconds = search_speed.seconds_to_depth(threads=2, fens=fens, depth=5, hash_mb=16)

    assert 0.0 <= seconds < 5.0
    received = log.read_text().splitlines()
    assert received[:3] == [
        "setoption name Threads value 2",
        "setoption name Hash value 16",
        "isready",
    ]
    assert received.count("go depth 5") == 2
    assert [line for line in received if line.startswith("position")] == [
        f"position fen {fen}" for fen in fens
    ]
    assert received[-1] == "quit"
