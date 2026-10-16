"""Tests for the parallel round-robin tournament."""

import importlib.util
import json
import math
import os
import zlib
from pathlib import Path

import chess
import pytest

from chess_engine.comparison import tournament
from chess_engine.comparison.tournament import (
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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
    # Nothing is recorded for the requeued game, so a rerun replays both games.
    assert read_records(out) == []
    monkeypatch.setattr(tournament, "MAX_POOL_REBUILDS", 5)
    summary = run_round_robin(["h2", "h3"], [opening], 2.0, 0.0, 1, out)
    assert summary["played"] == 2 and summary["errors"] == 0


def _dies_on_first_opening_as_h2(job):
    """Kill the worker when h2 plays White from the first opening of ``paired_openings(2, 2)``."""
    if job[0] == "h2" and job[2] == paired_openings(1, seed=2)[0]:
        os._exit(1)
    return tournament._play_game(job)


@pytest.mark.slow
def test_culprit_is_recorded_after_two_deaths_and_the_others_are_played(tmp_path):
    out = tmp_path / "games.jsonl"
    openings = paired_openings(2, seed=2)
    summary = run_round_robin(
        ["h2", "h3"],
        openings,
        base=2.0,
        increment=0.0,
        workers=2,
        out=out,
        play=_dies_on_first_opening_as_h2,
    )
    records = read_records(out)
    keys = [(r["white"], r["black"], r["fen"]) for r in records]
    expected = {
        ("h2", "h3", openings[0]),
        ("h3", "h2", openings[0]),
        ("h2", "h3", openings[1]),
        ("h3", "h2", openings[1]),
    }
    assert len(records) == 4 and set(keys) == expected  # nothing lost, nothing duplicated
    errors = [r for r in records if "error" in r]
    assert len(errors) == 1
    assert (errors[0]["white"], errors[0]["black"], errors[0]["fen"]) == (
        "h2",
        "h3",
        openings[0],
    )
    assert "died 2 times" in errors[0]["error"]
    assert all("winner" in r for r in records if "error" not in r)
    assert summary["errors"] == 1 and summary["played"] == 4


def _dies_on_every_game(job):
    """A play function whose worker always dies, so that no game ever finishes."""
    os._exit(1)


@pytest.mark.slow
def test_rebuild_limit_is_enforced_with_several_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(tournament, "MAX_POOL_REBUILDS", 1)
    monkeypatch.setattr(tournament, "MAX_DEATHS_PER_GAME", 3)
    out = tmp_path / "games.jsonl"
    openings = paired_openings(2, seed=2)
    with pytest.raises(RuntimeError, match="giving up"):
        run_round_robin(
            ["h2", "h3"],
            openings,
            base=2.0,
            increment=0.0,
            workers=2,
            out=out,
            play=_dies_on_every_game,
        )
    assert read_records(out) == []


def _dies_once_as_h2_white(job):
    """Kill the worker the first time h2 plays White from either opening; later attempts play.

    The first attempt is recorded by a marker file in ``$TOURNAMENT_TEST_MARKERS``, which the
    spawned workers inherit.
    """
    if job[0] == "h2" and job[2] in paired_openings(2, seed=2):
        marker = Path(os.environ["TOURNAMENT_TEST_MARKERS"]) / str(zlib.crc32(job[2].encode()))
        if not marker.exists():
            marker.touch()
            os._exit(1)
    return tournament._play_game(job)


@pytest.mark.slow
def test_rebuild_counter_resets_after_each_game_played(tmp_path, monkeypatch):
    """Two broken pools, one per opening, with a game played in between: the limit of 1 holds."""
    markers = tmp_path / "markers"
    markers.mkdir()
    monkeypatch.setenv("TOURNAMENT_TEST_MARKERS", str(markers))
    monkeypatch.setattr(tournament, "MAX_POOL_REBUILDS", 1)
    out = tmp_path / "games.jsonl"
    summary = run_round_robin(
        ["h2", "h3"],
        paired_openings(2, seed=2),
        base=2.0,
        increment=0.0,
        workers=1,
        out=out,
        play=_dies_once_as_h2_white,
    )
    records = read_records(out)
    assert len(records) == 4 and all("error" not in r for r in records)
    assert summary["errors"] == 0 and summary["played"] == 4


@pytest.mark.slow
def test_run_gives_up_when_every_game_dies(tmp_path, monkeypatch):
    """Error records do not reset the count, so a systemic failure ends the run."""
    monkeypatch.setattr(tournament, "MAX_POOL_REBUILDS", 2)
    out = tmp_path / "games.jsonl"
    with pytest.raises(RuntimeError, match="giving up"):
        run_round_robin(
            ["h2", "h3"],
            paired_openings(2, seed=2),
            base=2.0,
            increment=0.0,
            workers=2,
            out=out,
            play=_dies_on_every_game,
        )
    assert all("error" in r for r in read_records(out))


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
    assert h2["elo_vs_anchor"] == pytest.approx(-400 * math.log10(1 / 0.375 - 1))

    m = summary["players"]["m"]
    assert m["games"] == 4
    assert (m["wins"], m["draws"], m["losses"]) == (3, 0, 1)
    assert m["score"] == pytest.approx(75.0)
    assert m["elo_vs_anchor"] == pytest.approx(-400 * math.log10(1 / 0.75 - 1))

    assert summary["players"]["h3"]["elo_vs_anchor"] == pytest.approx(0.0)
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
    assert h2["elo"] == pytest.approx(h2["elo_vs_anchor"], abs=0.5)


def test_summarize_requires_the_anchor_player():
    with pytest.raises(ValueError):
        summarize([{"white": "h2", "black": "m", "winner": None}])


def test_size_for_budget():
    # 1 h, 2 workers, 10 s per game -> 720 games; 3 players -> 3 pairs -> 6 games per opening.
    assert size_for_budget(hours=1, workers=2, n_players=3, seconds_per_game=10) == 120
    # 10 h, 5 workers, 10 s per game -> 18000 games; 5 players -> 10 pairs -> 20 per opening.
    assert size_for_budget(hours=10, workers=5, n_players=5, seconds_per_game=10) == 900


def test_complete_record_without_newline_is_kept(tmp_path):
    out = tmp_path / "games.jsonl"
    opening = paired_openings(1, seed=2)[0]
    finished = {"white": "h2", "black": "h3", "fen": opening, "base": 2.0, "increment": 0.0}
    line = json.dumps(finished).encode()
    out.write_bytes(line)
    tournament._repair_tail(out)
    assert out.read_bytes() == line + b"\n"
    assert tournament.read_games(out) == [finished]


def test_truncated_multibyte_character_is_a_truncated_line(tmp_path, caplog):
    out = tmp_path / "games.jsonl"
    opening = paired_openings(1, seed=2)[0]
    finished = (
        json.dumps(
            {"white": "h2", "black": "h3", "fen": opening, "base": 2.0, "increment": 0.0}
        ).encode()
        + b"\n"
    )
    partial = '{"note": "\u00e9"'.encode()[:-2]  # cut inside the two-byte character
    out.write_bytes(finished + partial)
    assert len(tournament.read_games(out)) == 1
    assert "truncated" in caplog.text
    tournament._repair_tail(out)
    assert out.read_bytes() == finished


@pytest.fixture(scope="module")
def script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "bench" / "run_tournament.py"
    spec = importlib.util.spec_from_file_location("run_tournament", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_rejects_fewer_than_two_distinct_players(script, capsys):
    with pytest.raises(SystemExit):
        script.parse_args(["--players", "h3"])
    assert "at least two distinct players" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        script.parse_args(["--players", "h3,h3"])


def test_parse_args_requires_the_anchor(script, capsys):
    with pytest.raises(SystemExit):
        script.parse_args(["--players", "h2,m"])
    assert "must include h3" in capsys.readouterr().err
    assert "h3" in script.parse_args([]).specs


def test_calibrate_returns_none_when_no_game_is_played(script):
    assert script.calibrate(["h3"], 2.0, 0.0, 1) is None


def test_tournament_accepts_tagged_players_and_custom_anchor(script, tmp_path, monkeypatch, capsys):
    from chess_engine.models import base as models_base
    from chess_engine.models import registry
    from chess_engine.models.linear_eval import PstTaperedModel

    monkeypatch.setattr(models_base, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(registry, "ARTIFACTS_DIR", tmp_path)
    PstTaperedModel().save(tmp_path / "pst_tapered@x.npz")
    tournament.check_specs(["h2", "pst_tapered@x"])
    with pytest.raises(ValueError, match="pst_tapered@missing"):
        tournament.check_specs(["pst_tapered@missing"])

    args = script.parse_args(["--players", "pst_tapered@x,m", "--anchor", "m"])
    assert args.specs == ["pst_tapered@x", "m"] and args.anchor == "m"
    with pytest.raises(SystemExit):
        script.parse_args(["--players", "h2,m", "--anchor", "h3"])
    assert "must include h3" in capsys.readouterr().err

    results = [game("pst_tapered@x", "m", "white"), game("m", "pst_tapered@x", "white")]
    summary = summarize(results, anchor="m")
    assert summary["players"]["m"]["elo"] == pytest.approx(0.0)
    assert summary["players"]["m"]["elo_vs_anchor"] == pytest.approx(0.0)
    assert summary["players"]["pst_tapered@x"]["elo_vs_anchor"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="'m'"):
        summarize([game("h2", "h3", "white")], anchor="m")


def test_record_cards_writes_tournament_sections_shown_by_the_history(tmp_path, monkeypatch):
    from chess_engine.comparison import report
    from chess_engine.models import cards

    monkeypatch.setattr(cards, "CARDS_DIR", tmp_path / "cards")
    cards.update_card("m", status="current", architecture="A")
    results = [
        game("m", "h3", "white"),
        game("h3", "m", "white"),
        game("m", "h2", None),
        game("h2", "m", "black"),
        game("h3", "h2", "white"),
        game("m2", "h3", "black"),
    ]
    summary = summarize(results)
    assert summary["players"]["m"]["elo_interval"] > 0

    recorded = tournament.record_cards(summary, ["h2", "h3", "m", "m2"], "h3", 10.0, 0.1)
    assert recorded == ["m"]
    section = cards.load_card("m")["tournament"]
    entry = summary["players"]["m"]
    assert section == {
        "games": entry["games"],
        "score": entry["score"],
        "elo": entry["elo"],
        "elo_interval": entry["elo_interval"],
        "anchor": "h3",
        "base": 10.0,
        "increment": 0.1,
    }
    assert not cards.card_path("m2").exists() and not cards.card_path("h3").exists()
    row = next(line for line in report.render_tables().splitlines() if "`m`" in line)
    assert f"{entry['score']:.1f}%" in row and "±" in row
