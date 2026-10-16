"""Tests for the parallel round-robin tournament."""

import json
import math
import os

import chess
import pytest

from chess_engine.training import tournament
from chess_engine.training.tournament import (
    paired_openings,
    run_round_robin,
    size_for_budget,
    summarize,
)


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_paired_openings_are_distinct_and_playable():
    openings = paired_openings(5, plies=6, seed=1)
    assert len(openings) == 5
    assert len(set(openings)) == 5
    for fen in openings:
        board = chess.Board(fen)
        assert board.is_valid()
        assert not board.is_game_over()
    assert paired_openings(5, plies=6, seed=1) == openings


def test_every_pair_plays_both_colours(tmp_path):
    out = tmp_path / "games.jsonl"
    opening = paired_openings(1, seed=2)[0]
    run_round_robin(["h2", "h3"], [opening], base=2.0, increment=0.0, workers=2, out=out)
    records = read_records(out)
    assert len(records) == 2
    assert {(r["white"], r["black"]) for r in records} == {("h2", "h3"), ("h3", "h2")}
    assert all(r["fen"] == opening and "error" not in r for r in records)
    assert all(r["base"] == 2.0 and r["increment"] == 0.0 for r in records)


def test_resume_skips_finished_games(tmp_path):
    out = tmp_path / "games.jsonl"
    opening = paired_openings(1, seed=2)[0]
    finished = {
        "white": "h2",
        "black": "h3",
        "fen": opening,
        "base": 2.0,
        "increment": 0.0,
        "winner": "white",
        "result": "1-0",
        "termination": "checkmate",
        "total_moves": 10,
    }
    out.write_text(json.dumps(finished) + "\n")

    summary = run_round_robin(["h2", "h3"], [opening], base=2.0, increment=0.0, workers=2, out=out)

    records = read_records(out)
    assert len(records) == 2
    assert records[0] == finished
    assert (records[1]["white"], records[1]["black"]) == ("h3", "h2")
    assert summary["played"] == 1 and summary["skipped"] == 1


def test_crashed_game_is_recorded_not_fatal(tmp_path, monkeypatch):
    # Skip the up-front validation so that the bad spec only fails inside the worker.
    monkeypatch.setattr(tournament, "check_specs", lambda specs: None)
    out = tmp_path / "games.jsonl"
    opening = paired_openings(1, seed=2)[0]
    summary = run_round_robin(
        ["h3", "no_such_model"], [opening], base=2.0, increment=0.0, workers=2, out=out
    )
    records = read_records(out)
    assert len(records) == 2
    assert all("KeyError" in r["error"] for r in records)
    assert summary["errors"] == 2
    assert summarize(records)["players"]["h3"]["games"] == 0


def test_unknown_player_fails_before_any_game(tmp_path):
    out = tmp_path / "games.jsonl"
    opening = paired_openings(1, seed=2)[0]
    with pytest.raises(ValueError, match="no_such_model"):
        run_round_robin(
            ["h2", "no_such_model"], [opening], base=2.0, increment=0.0, workers=2, out=out
        )
    assert not out.exists()


def _dies_on_first_white_h2(job):
    """Play like the real worker, except that the worker process dies when h2 is White."""
    if job[0] == "h2":
        os._exit(1)
    return tournament._play_game(job)


def test_dead_worker_is_recorded_and_pool_rebuilt(tmp_path):
    out = tmp_path / "games.jsonl"
    opening = paired_openings(1, seed=2)[0]
    summary = run_round_robin(
        ["h2", "h3"],
        [opening],
        base=2.0,
        increment=0.0,
        workers=1,
        out=out,
        play=_dies_on_first_white_h2,
    )
    records = read_records(out)
    assert len(records) == 2
    assert [("error" in r) for r in records] == [True, False]
    assert "died" in records[0]["error"]
    assert records[1]["winner"] in ("white", "black", None)
    assert summary["errors"] == 1 and summary["played"] == 2


def test_run_gives_up_after_too_many_pool_rebuilds(tmp_path, monkeypatch):
    monkeypatch.setattr(tournament, "MAX_POOL_REBUILDS", 0)
    out = tmp_path / "games.jsonl"
    opening = paired_openings(1, seed=2)[0]
    with pytest.raises(RuntimeError, match="giving up"):
        run_round_robin(
            ["h2", "h3"],
            [opening],
            base=2.0,
            increment=0.0,
            workers=1,
            out=out,
            play=_dies_on_first_white_h2,
        )
    records = read_records(out)
    assert len(records) == 1 and "died" in records[0]["error"]


def test_truncated_last_line_is_ignored_and_repaired(tmp_path, caplog):
    out = tmp_path / "games.jsonl"
    opening = paired_openings(1, seed=2)[0]
    finished = {
        "white": "h2",
        "black": "h3",
        "fen": opening,
        "base": 2.0,
        "increment": 0.0,
        "winner": "white",
    }
    out.write_text(json.dumps(finished) + "\n" + '{"white": "h3", "bla')
    assert len(tournament.read_games(out)) == 1
    assert "truncated" in caplog.text

    summary = run_round_robin(["h2", "h3"], [opening], 2.0, 0.0, 2, out)

    lines = out.read_text().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line) for line in lines)
    assert summary["played"] == 1 and summary["games"] == 2


def test_corrupt_line_in_the_middle_raises(tmp_path):
    out = tmp_path / "games.jsonl"
    out.write_text('{"white": "h2"\n{"white": "h3", "black": "h2"}\n')
    with pytest.raises(ValueError, match="line 1"):
        tournament.read_games(out)


def test_other_time_control_is_refused(tmp_path):
    out = tmp_path / "games.jsonl"
    opening = paired_openings(1, seed=2)[0]
    stale = {"white": "h2", "black": "h3", "fen": opening, "base": 10.0, "increment": 0.1}
    out.write_text(json.dumps(stale) + "\n")
    with pytest.raises(ValueError, match="another time control"):
        run_round_robin(["h2", "h3"], [opening], base=2.0, increment=0.0, workers=2, out=out)
    assert read_records(out) == [stale]


def game(white, black, winner, **extra):
    return {"white": white, "black": black, "winner": winner, **extra}


def test_summarize_scores_and_elo():
    results = [
        game("h2", "h3", "black"),  # h3 wins
        game("h3", "h2", "white"),  # h3 wins
        game("h2", "h3", None),  # draw
        game("h3", "h2", "black"),  # h2 wins
        game("m", "h3", "white"),  # m wins
        game("h3", "m", "white"),  # h3 wins
        game("m", "h3", "white"),  # m wins
        game("h3", "m", "black"),  # m wins
        game("m", "h2", None, error="KeyError: boom"),  # excluded from every score
    ]
    summary = summarize(results)
    h2 = summary["players"]["h2"]
    assert h2["games"] == 4
    assert (h2["wins"], h2["draws"], h2["losses"]) == (1, 1, 2)
    assert h2["score"] == pytest.approx(37.5)
    assert h2["elo_vs_h3"] == pytest.approx(-400 * math.log10(1 / 0.375 - 1))

    m = summary["players"]["m"]
    assert m["games"] == 4
    assert (m["wins"], m["draws"], m["losses"]) == (3, 0, 1)
    assert m["score"] == pytest.approx(75.0)
    assert m["elo_vs_h3"] == pytest.approx(-400 * math.log10(1 / 0.75 - 1))

    assert summary["players"]["h3"]["elo_vs_h3"] == pytest.approx(0.0)
    lo, hi = h2["ci95"]
    assert lo < 37.5 < hi


def test_iterative_fit_matches_direct_score_for_two_players():
    results = [
        game("h2", "h3", "black"),
        game("h3", "h2", "white"),
        game("h2", "h3", None),
        game("h3", "h2", "black"),
    ]
    h2 = summarize(results)["players"]["h2"]
    assert h2["elo"] == pytest.approx(h2["elo_vs_h3"], abs=0.5)


def test_summarize_requires_the_anchor_player():
    with pytest.raises(ValueError):
        summarize([{"white": "h2", "black": "m", "winner": None}])


def test_size_for_budget():
    # 1 h, 2 workers, 10 s per game -> 720 games; 3 players -> 3 pairs -> 6 games per opening.
    assert size_for_budget(hours=1, workers=2, n_players=3, seconds_per_game=10) == 120
    # 10 h, 5 workers, 10 s per game -> 18000 games; 5 players -> 10 pairs -> 20 per opening.
    assert size_for_budget(hours=10, workers=5, n_players=5, seconds_per_game=10) == 900
