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


def test_unknown_player_fails_before_any_game(tmp_path):
    out = tmp_path / "games.jsonl"
    opening = paired_openings(1, seed=2)[0]
    with pytest.raises(ValueError, match="no_such_model"):
        run_round_robin(
            ["mlp_features", "no_such_model"],
            [opening],
            base=2.0,
            increment=0.0,
            workers=2,
            out=out,
        )
    assert not out.exists()


def _dies_on_first_white_mlp(job):
    """Play like the real worker, except that the worker process dies when mlp_features is White."""
    if job[0] == "mlp_features":
        os._exit(1)
    return tournament._play_game(job)


def _dies_on_first_opening_as_mlp(job):
    """Kill the worker when mlp_features plays White from the first opening of ``paired_openings(2, 2)``."""
    if job[0] == "mlp_features" and job[2] == paired_openings(1, seed=2)[0]:
        os._exit(1)
    return tournament._play_game(job)


def _dies_on_every_game(job):
    """A play function whose worker always dies, so that no game ever finishes."""
    os._exit(1)


def _dies_once_as_mlp_white(job):
    """Kill the worker the first time mlp_features plays White from either opening; later attempts play.

    The first attempt is recorded by a marker file in ``$TOURNAMENT_TEST_MARKERS``, which the
    spawned workers inherit.
    """
    if job[0] == "mlp_features" and job[2] in paired_openings(2, seed=2):
        marker = Path(os.environ["TOURNAMENT_TEST_MARKERS"]) / str(zlib.crc32(job[2].encode()))
        if not marker.exists():
            marker.touch()
            os._exit(1)
    return tournament._play_game(job)


def test_corrupt_line_in_the_middle_raises(tmp_path):
    out = tmp_path / "games.jsonl"
    out.write_text('{"white": "mlp_features"\n{"white": "h1", "black": "mlp_features"}\n')
    with pytest.raises(ValueError, match="line 1"):
        tournament.read_games(out)


def test_other_time_control_is_refused(tmp_path):
    out = tmp_path / "games.jsonl"
    opening = paired_openings(1, seed=2)[0]
    stale = {"white": "mlp_features", "black": "h1", "fen": opening, "base": 10.0, "increment": 0.1}
    out.write_text(json.dumps(stale) + "\n")
    with pytest.raises(ValueError, match="another time control"):
        run_round_robin(
            ["mlp_features", "h1"], [opening], base=2.0, increment=0.0, workers=2, out=out
        )
    assert read_records(out) == [stale]


def game(white, black, winner, **extra):
    return {"white": white, "black": black, "winner": winner, **extra}


def test_summarize_scores_and_elo():
    results = [
        game("mlp_features", "h1", "black"),  # h1 wins
        game("h1", "mlp_features", "white"),  # h1 wins
        game("mlp_features", "h1", None),  # draw
        game("h1", "mlp_features", "black"),  # mlp_features wins
        game("m", "h1", "white"),  # m wins
        game("h1", "m", "white"),  # h1 wins
        game("m", "h1", "white"),  # m wins
        game("h1", "m", "black"),  # m wins
        game("m", "mlp_features", None, error="KeyError: boom"),  # excluded from every score
    ]
    summary = summarize(results, anchor="h1")
    mlp = summary["players"]["mlp_features"]
    assert mlp["games"] == 4
    assert (mlp["wins"], mlp["draws"], mlp["losses"]) == (1, 1, 2)
    assert mlp["score"] == pytest.approx(37.5)
    assert mlp["elo_vs_anchor"] == pytest.approx(-400 * math.log10(1 / 0.375 - 1))

    m = summary["players"]["m"]
    assert m["games"] == 4
    assert (m["wins"], m["draws"], m["losses"]) == (3, 0, 1)
    assert m["score"] == pytest.approx(75.0)
    assert m["elo_vs_anchor"] == pytest.approx(-400 * math.log10(1 / 0.75 - 1))

    assert summary["players"]["h1"]["elo_vs_anchor"] == pytest.approx(0.0)
    lo, hi = mlp["ci95"]
    assert lo < 37.5 < hi


def test_iterative_fit_matches_direct_score_for_two_players():
    results = [
        game("mlp_features", "h1", "black"),
        game("h1", "mlp_features", "white"),
        game("mlp_features", "h1", None),
        game("h1", "mlp_features", "black"),
    ]
    mlp = summarize(results, anchor="h1")["players"]["mlp_features"]
    assert mlp["elo"] == pytest.approx(mlp["elo_vs_anchor"], abs=0.5)


def test_summarize_requires_the_anchor_player():
    with pytest.raises(ValueError):
        summarize([{"white": "mlp_features", "black": "m", "winner": None}])


def test_size_for_budget():
    # 1 h, 2 workers, 10 s per game -> 720 games; 3 players -> 3 pairs -> 6 games per opening.
    assert size_for_budget(hours=1, workers=2, n_players=3, seconds_per_game=10) == 120
    # 10 h, 5 workers, 10 s per game -> 18000 games; 5 players -> 10 pairs -> 20 per opening.
    assert size_for_budget(hours=10, workers=5, n_players=5, seconds_per_game=10) == 900


def test_complete_record_without_newline_is_kept(tmp_path):
    out = tmp_path / "games.jsonl"
    opening = paired_openings(1, seed=2)[0]
    finished = {
        "white": "mlp_features",
        "black": "h1",
        "fen": opening,
        "base": 2.0,
        "increment": 0.0,
    }
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
            {"white": "mlp_features", "black": "h1", "fen": opening, "base": 2.0, "increment": 0.0}
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
        script.parse_args(["--players", "h1"])
    assert "at least two distinct players" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        script.parse_args(["--players", "h1,h1"])


def test_parse_args_requires_the_anchor(script, capsys):
    with pytest.raises(SystemExit):
        script.parse_args(["--players", "mlp_features,m"])
    assert "must include h1" in capsys.readouterr().err
    assert "h1" in script.parse_args([]).specs


def test_calibrate_returns_none_when_no_game_is_played(script):
    assert script.calibrate(["h1"], 2.0, 0.0, 1) is None


def test_tournament_accepts_tagged_players_and_custom_anchor(script, tmp_path, monkeypatch, capsys):
    from chess_engine.models import base as models_base
    from chess_engine.models import registry
    from chess_engine.models.linear_eval import PstTaperedModel

    monkeypatch.setattr(models_base, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(registry, "ARTIFACTS_DIR", tmp_path)
    PstTaperedModel().save(tmp_path / "linear_pst@x.npz")
    tournament.check_specs(["h1", "linear_pst@x"])
    with pytest.raises(ValueError, match="linear_pst@missing"):
        tournament.check_specs(["linear_pst@missing"])

    args = script.parse_args(["--players", "linear_pst@x,m", "--anchor", "m"])
    assert args.specs == ["linear_pst@x", "m"] and args.anchor == "m"
    with pytest.raises(SystemExit):
        script.parse_args(["--players", "mlp_features,m", "--anchor", "h1"])
    assert "must include h1" in capsys.readouterr().err

    results = [game("linear_pst@x", "m", "white"), game("m", "linear_pst@x", "white")]
    summary = summarize(results, anchor="m")
    assert summary["players"]["m"]["elo"] == pytest.approx(0.0)
    assert summary["players"]["m"]["elo_vs_anchor"] == pytest.approx(0.0)
    assert summary["players"]["linear_pst@x"]["elo_vs_anchor"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="'m'"):
        summarize([game("mlp_features", "h1", "white")], anchor="m")


def test_record_cards_writes_tournament_sections_shown_by_the_history(tmp_path, monkeypatch):
    from chess_engine.comparison import report
    from chess_engine.models import cards

    monkeypatch.setattr(cards, "CARDS_DIR", tmp_path / "cards")
    cards.update_card("m", status="current", architecture="A")
    results = [
        game("m", "h1", "white"),
        game("h1", "m", "white"),
        game("m", "mlp_features", None),
        game("mlp_features", "m", "black"),
        game("h1", "mlp_features", "white"),
        game("m2", "h1", "black"),
    ]
    summary = summarize(results, anchor="h1")
    assert summary["players"]["m"]["elo_interval"] > 0

    recorded = tournament.record_cards(summary, ["mlp_features", "h1", "m", "m2"], "h1", 10.0, 0.1)
    assert recorded == ["m"]
    section = cards.load_card("m")["tournament"]
    entry = summary["players"]["m"]
    assert section == {
        "games": entry["games"],
        "score": entry["score"],
        "elo": entry["elo"],
        "elo_interval": entry["elo_interval"],
        "anchor": "h1",
        "base": 10.0,
        "increment": 0.1,
    }
    assert not cards.card_path("m2").exists() and not cards.card_path("h1").exists()
    row = next(line for line in report.render_tables().splitlines() if "`m`" in line)
    assert f"{entry['score']:.1f}%" in row and "±" in row
