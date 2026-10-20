"""Game loop, results, standings, PGN export and configuration loading."""

import io
import json
import time

import chess
import chess.pgn
import pytest

from chess_engine.bots.chess_bot import ChessBot
from chess_engine.engine.data_classes import GameResult, TournamentConfig, TournamentResult
from chess_engine.engine.game_manager import GameManager, load_config

SCHOLARS_MATE_WHITE = ["e2e4", "f1c4", "d1h5", "h5f7"]
SCHOLARS_MATE_BLACK = ["e7e5", "b8c6", "g8f6"]


class ScriptedBot(ChessBot):
    """Plays a fixed list of UCI moves, one per call, and records the game hooks."""

    def __init__(self, name, moves, delay=0.0, fail=False):
        super().__init__(name, "tests")
        self.moves = list(moves)
        self.delay = delay
        self.fail = fail
        self.events = []

    def on_game_start(self, color, opponent_name):
        self.events.append(("start", color, opponent_name))

    def on_game_end(self, result, board):
        self.events.append(("end", result))

    def get_move(self, board, time_limit):
        if self.fail:
            raise RuntimeError("boom")
        time.sleep(self.delay)
        return chess.Move.from_uci(self.moves.pop(0))


class FakeRegistry:
    """Minimal stand-in for BotRegistry: maps names to ready-made bots."""

    def __init__(self, bots):
        self.bots = bots

    def get_bot(self, name):
        return self.bots.get(name)


@pytest.fixture
def manager():
    return GameManager(FakeRegistry({}))


def scholars_mate(manager, **kwargs):
    white = ScriptedBot("White", SCHOLARS_MATE_WHITE)
    black = ScriptedBot("Black", SCHOLARS_MATE_BLACK)
    return white, black, manager.play_game(white, black, **kwargs)


class TestPlayGame:
    def test_checkmate_is_scored_and_reported(self, manager):
        white, black, result = scholars_mate(manager)
        assert (result.result, result.winner, result.termination) == ("1-0", "white", "checkmate")
        assert result.moves == ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
        assert result.total_moves == 7 and chess.Board(result.final_fen).is_checkmate()
        assert manager.completed_games == [result]

    def test_players_are_told_when_the_game_starts_and_ends(self, manager):
        white, black, _ = scholars_mate(manager)
        assert white.events == [("start", chess.WHITE, "Black"), ("end", "1-0")]
        assert black.events[0] == ("start", chess.BLACK, "White")

    def test_an_illegal_move_loses_the_game(self, manager):
        white, black = ScriptedBot("W", ["e2e5"]), ScriptedBot("B", [])
        result = manager.play_game(white, black)
        assert (result.winner, result.termination, result.result) == (
            "black",
            "illegal_move",
            "0-1",
        )
        assert result.moves == []

    def test_an_exception_in_a_bot_loses_the_game(self, manager):
        result = manager.play_game(
            ScriptedBot("W", SCHOLARS_MATE_WHITE), ScriptedBot("B", [], fail=True)
        )
        assert (result.winner, result.termination, result.result) == (
            "white",
            "player_error",
            "1-0",
        )

    def test_exceeding_the_clock_loses_on_time(self, manager):
        slow = ScriptedBot("W", ["e2e4"], delay=0.2)
        result = manager.play_game(slow, ScriptedBot("B", ["e7e5"]), time_limit=0.05)
        assert (result.winner, result.termination, result.result) == (
            "black",
            "time_forfeit",
            "0-1",
        )

    def test_a_position_that_is_already_drawn_ends_immediately(self, manager):
        result = manager.play_game(
            ScriptedBot("W", []),
            ScriptedBot("B", []),
            starting_position="k7/8/8/8/8/8/8/K7 w - - 0 1",
        )
        assert (result.result, result.winner, result.termination) == (
            "1/2-1/2",
            None,
            "insufficient_material",
        )
        assert result.total_moves == 0

    def test_increment_is_added_after_every_move(self, manager):
        _, _, result = scholars_mate(manager, time_limit=30.0, time_increment=2.0)
        assert result.white_time_left > 30.0 and result.black_time_left > 30.0

    def test_games_can_start_from_a_custom_position(self, manager):
        fen = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"
        result = manager.play_game(
            ScriptedBot("W", ["a1a8"]), ScriptedBot("B", []), starting_position=fen
        )
        assert result.starting_fen.startswith("6k1/5ppp") and result.termination == "checkmate"

    def test_verbose_mode_prints_the_game(self, manager, capsys):
        scholars_mate(manager, verbose=True)
        output = capsys.readouterr().out
        assert "Starting game" in output and "Qxf7#" in output


class TestPgn:
    def test_pgn_is_valid_and_replays_the_game(self, manager):
        _, _, result = scholars_mate(manager)
        game = chess.pgn.read_game(io.StringIO(result.to_pgn(time_limit=60, time_increment=1)))
        assert game.headers["White"] == "White" and game.headers["Result"] == "1-0"
        assert game.end().board().is_checkmate()

    def test_forfeit_results_are_recorded_in_the_pgn(self, manager):
        result = manager.play_game(ScriptedBot("W", ["e2e5"]), ScriptedBot("B", []))
        assert chess.pgn.read_game(io.StringIO(result.to_pgn())).headers["Result"] == "0-1"


def finished_game(white, black, winner, round_number=None):
    return GameResult(
        white,
        black,
        f"{white}-{black}",
        "t",
        "*",
        winner,
        "checkmate",
        [],
        0,
        0.0,
        chess.STARTING_FEN,
        chess.STARTING_FEN,
        0.0,
        0.0,
        round_number,
    )


class TestTournamentResults:
    def results(self):
        games = [
            finished_game("A", "B", "white", 1),
            finished_game("B", "A", None, 1),
            finished_game("A", "C", "black", 1),
            finished_game("C", "A", "white", 1),
            finished_game("B", "C", None, 1),
            finished_game("C", "B", "black", 1),
        ]
        return TournamentResult(
            "id", "now", TournamentConfig(60, 0, 1), ["A", "B", "C"], games, 1.0
        )

    def test_standings_use_the_scoring_system_and_sort_by_score_then_wins(self):
        standings = dict(self.results().get_standings())
        assert standings["A"] == {
            "score": 1.5,
            "wins": 1,
            "draws": 1,
            "losses": 2,
            "games_played": 4,
        }
        assert standings["B"]["score"] == 2.0 and standings["C"]["score"] == 2.5
        assert [name for name, _ in self.results().get_standings()] == ["C", "B", "A"]

    def test_custom_scoring_is_honoured(self):
        result = self.results()
        result.config = TournamentConfig(60, 0, 1, {"win": 3, "draw": 1, "loss": 0})
        assert dict(result.get_standings())["A"]["score"] == 4

    def test_results_are_saved_as_json(self, tmp_path):
        path = tmp_path / "t.json"
        self.results().save_to_file(str(path))
        data = json.loads(path.read_text())
        assert data["bot_names"] == ["A", "B", "C"] and len(data["games"]) == 6

    def test_games_are_exported_as_one_pgn_each(self, tmp_path):
        result = self.results()
        result.games[0] = GameResult(
            *[
                getattr(result.games[0], f)
                for f in ("white_name", "black_name", "game_id", "timestamp")
            ],
            "1-0",
            "white",
            "checkmate",
            ["e2e4"],
            1,
            0.0,
            chess.STARTING_FEN,
            chess.Board().fen(),
            0.0,
            0.0,
            1,
        )
        result.export_games_to_pgn_directory(str(tmp_path / "pgn"))
        assert (tmp_path / "pgn" / "A-B-1.pgn").exists()


class TestTournament:
    def test_round_robin_plays_every_ordered_pair(self):
        def fool(name):
            return ScriptedBot(name, ["f2f3", "g2g4", "e7e5", "d8h4"])

        # Each bot is scripted for both colours: as White f3, g4; as Black e5, Qh4#.
        class FoolBot(ScriptedBot):
            def on_game_start(self, color, opponent_name):
                self.moves = ["f2f3", "g2g4"] if color == chess.WHITE else ["e7e5", "d8h4"]

        registry = FakeRegistry({"X": FoolBot("X", []), "Y": FoolBot("Y", [])})
        result = GameManager(registry).run_tournament(["X", "Y"], TournamentConfig(30, 0, 2))
        assert len(result.games) == 4
        assert {game.winner for game in result.games} == {"black"}
        standings = dict(result.get_standings())
        assert standings["X"]["score"] == standings["Y"]["score"] == 2.0

    def test_unknown_bots_and_too_few_bots_are_rejected(self):
        manager = GameManager(FakeRegistry({}))
        with pytest.raises(ValueError, match="at least 2"):
            manager.run_tournament(["X"], TournamentConfig(30, 0, 1))
        with pytest.raises(ValueError, match="not found"):
            manager.run_tournament(["X", "Y"], TournamentConfig(30, 0, 1))


def test_configuration_is_loaded_from_json(tmp_path):
    path = tmp_path / "tournament.json"
    path.write_text(json.dumps({"total_game_time": 90, "time_increment": 1, "rounds": 3}))
    config = load_config(str(path))
    assert (config.total_game_time, config.time_increment, config.rounds) == (90, 1, 3)
    assert config.scoring == {"win": 1.0, "draw": 0.5, "loss": 0.0}


def test_missing_configuration_file_falls_back_to_the_default_preset(tmp_path):
    config = load_config(str(tmp_path / "missing.json"))
    assert (config.total_game_time, config.time_increment, config.rounds) == (180.0, 2.0, 1)
