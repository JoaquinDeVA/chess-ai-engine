"""Tests for the parallel round-robin tournament."""

import json
import os
import zlib
from pathlib import Path

import chess
import pytest

from chess_engine import _core
from chess_engine.comparison import play_tournament, tournament
from chess_engine.comparison.standings import summarize
from chess_engine.comparison.tournament import paired_openings, run_round_robin


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
            ["linear_material", "no_such_model"],
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
    return play_tournament


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


def test_parse_args_takes_an_explicit_number_of_openings(script):
    assert script.parse_args(["--openings", "1"]).openings == 1
    assert script.parse_args([]).openings is None


def test_calibrate_returns_none_when_no_game_is_played(script):
    assert script.calibrate(["h1"], 2.0, 0.0, 1, max_depth=1) is None


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


class _FakeStats:
    completed_depth = 3
    nodes_visited = 10
    max_depth_reached = 3
    iterations = ((1, 0.001, 3, False), (2, 0.01, 7, False))
    stop_reason = "estimate"
    stop_predicted = 12.5


class _FakePlayer:
    turn_budget = 0.5
    last_search_stats = _FakeStats()

    def get_move(self, board, time_left):
        return next(iter(board.legal_moves))


class _RaisingPlayer:
    def get_move(self, board, time_left):
        raise RuntimeError("fallo")


def test_recording_player_records_one_entry_per_move():
    board = chess.Board()
    fen = board.fen()
    player = tournament.RecordingPlayer(_FakePlayer())
    move = player.get_move(board, 9.0)
    assert len(player.moves) == 1
    entry = player.moves[0]
    assert entry.pop("move_seconds") >= 0
    assert entry == {
        "ply": 0,
        "color": "white",
        "fen": fen,
        "move": move.uci(),
        "time_left_before": 9.0,
        "budget": 0.5,
        "completed_depth": 3,
        "nodes": 10,
        "max_depth_reached": 3,
        "iterations": [[1, 0.001, 3, False], [2, 0.01, 7, False]],
        "stop_reason": "estimate",
        "stop_predicted": 12.5,
    }
    assert player.turn_budget == 0.5  # lo demas se delega


def test_recording_player_without_stats_uses_none():
    class Plain:
        def get_move(self, board, time_left):
            return next(iter(board.legal_moves))

    player = tournament.RecordingPlayer(Plain())
    player.get_move(chess.Board(), 1.0)
    entry = player.moves[0]
    assert [entry[k] for k in ("budget", "completed_depth", "nodes", "max_depth_reached")] == [
        None
    ] * 4


def test_recording_player_propagates_exceptions():
    player = tournament.RecordingPlayer(_RaisingPlayer())
    with pytest.raises(RuntimeError, match="fallo"):
        player.get_move(chess.Board(), 1.0)
    assert player.moves == []


def test_log_path_is_filename_safe(tmp_path):
    path = tournament.log_path(tmp_path, 7, "mlp:a/b", "h1")
    assert path == tmp_path / "7_mlp_a_b_vs_h1.json"


def test_play_game_writes_the_move_log(tmp_path, monkeypatch):
    class Result:
        winner, result, termination, total_moves = "white", "1-0", "checkmate", 1

    class FakeManager:
        def __init__(self, registry):
            pass

        def play_game(self, white, black, base, increment, **kwargs):
            white.get_move(chess.Board(kwargs["starting_position"]), base)
            return Result()

    depths = []

    def fake_make_player(spec, tm, max_depth=30, **options):
        depths.append(max_depth)
        return _FakePlayer()

    monkeypatch.setattr(tournament, "make_player", fake_make_player)
    monkeypatch.setattr(tournament, "GameManager", FakeManager)
    job = ("a:x", "h1", chess.STARTING_FEN, 5.0, 0.1, tmp_path / "logs", 3, 2)
    record = tournament._play_game(job)
    assert depths == [2, 2] and record["max_depth"] == 2
    saved = json.loads((tmp_path / "logs" / "3_a_x_vs_h1.json").read_text())
    assert {k: saved[k] for k in record} == record
    assert len(saved["moves"]) == 1 and saved["moves"][0]["ply"] == 0


def test_a_file_played_with_another_estimator_is_refused(tmp_path):
    out = tmp_path / "games.jsonl"
    opening = paired_openings(1, seed=2)[0]
    out.write_text(
        json.dumps({"white": "a", "black": "b", "fen": opening, "base": 2.0, "increment": 0.0})
        + "\n"
    )
    with pytest.raises(ValueError, match="estimator"):
        run_round_robin(
            ["linear_material", "h1"],
            [opening],
            base=2.0,
            increment=0.0,
            workers=1,
            out=out,
            estimator="geometric_growth",
        )


def test_records_without_the_field_count_as_the_original_estimator():
    tournament._check_estimator([{"white": "a"}], "ratio_squared", Path("x.jsonl"))
    with pytest.raises(ValueError, match="ratio_squared"):
        tournament._check_estimator([{"white": "a"}], "geometric_growth", Path("x.jsonl"))


def test_play_game_records_and_passes_the_estimator(monkeypatch):
    seen = []

    class Result:
        winner, result, termination, total_moves = None, "1/2-1/2", "stalemate", 0

    class FakeManager:
        def __init__(self, registry):
            pass

        def play_game(self, white, black, base, increment, **kwargs):
            return Result()

    def fake_make_player(spec, tm, max_depth=30, **options):
        seen.append(options)
        return _FakePlayer()

    monkeypatch.setattr(tournament, "make_player", fake_make_player)
    monkeypatch.setattr(tournament, "GameManager", FakeManager)
    record = tournament._play_game(
        ("a", "b", chess.STARTING_FEN, 5.0, 0.1, None, 0, 30, "node_growth")
    )
    assert record["estimator"] == "node_growth" and seen == [{"estimator": "node_growth"}] * 2
    default = tournament._play_game(("a", "b", chess.STARTING_FEN, 5.0, 0.1, None, 0, 30, None))
    assert default["estimator"] == _core.DEFAULT_ESTIMATOR


def test_records_without_a_core_count_as_the_python_engine():
    tournament._check_core([{"white": "a"}], "python", Path("x.jsonl"))
    with pytest.raises(ValueError, match="'python'.*'2.0.0'.*--out"):
        tournament._check_core([{"white": "a"}], "2.0.0", Path("x.jsonl"))


def test_games_of_the_same_core_and_an_empty_file_pass_the_core_check():
    tournament._check_core([{"white": "a", "core": "2.0.0"}], "2.0.0", Path("x.jsonl"))
    tournament._check_core([], "2.0.0", Path("x.jsonl"))
    with pytest.raises(ValueError, match="'1.9.0'"):
        tournament._check_core([{"core": "1.9.0"}], "2.0.0", Path("x.jsonl"))


def test_a_file_played_by_another_core_is_refused(tmp_path):
    out = tmp_path / "games.jsonl"
    opening = paired_openings(1, seed=2)[0]
    out.write_text(
        json.dumps(
            {
                "white": "a",
                "black": "b",
                "fen": opening,
                "base": 2.0,
                "increment": 0.0,
                "estimator": _core.DEFAULT_ESTIMATOR,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="core"):
        run_round_robin(
            ["linear_material", "h1"], [opening], base=2.0, increment=0.0, workers=1, out=out
        )


def test_play_game_records_the_core_version(monkeypatch):
    record = tournament._play_game(("a", "b", chess.STARTING_FEN, 5.0, 0.1))
    assert record["core"] == _core.version()
    assert tournament._error_record(("a", "b", "fen", 5.0, 0.1), "boom")["core"] == _core.version()
