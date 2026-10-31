"""Game loop of ``play_game``: results, clocks and forfeits (the clock is faked)."""

from types import SimpleNamespace

import chess

from chess_engine.game import play as play_module
from chess_engine.game.play import play_game

SCHOLARS_MATE_WHITE = ["e2e4", "f1c4", "d1h5", "h5f7"]
SCHOLARS_MATE_BLACK = ["e7e5", "b8c6", "g8f6"]


class FakeClock:
    """A clock that only moves when a bot "thinks": no real waiting in the tests."""

    def __init__(self):
        self.now = 1_000.0

    def time(self):
        return self.now


class ScriptedBot:
    """Plays a fixed list of UCI moves, one per call, and records the game hooks."""

    def __init__(self, name, moves, delay=0.0, fail=False, clock=None):
        self.name = name
        self.moves = list(moves)
        self.delay = delay
        self.clock = clock
        self.fail = fail
        self.events = []

    def on_game_start(self, color, opponent_name):
        self.events.append(("start", color, opponent_name))

    def get_move(self, board, time_limit):
        if self.fail:
            raise RuntimeError("boom")
        if self.clock:
            self.clock.now += self.delay
        return chess.Move.from_uci(self.moves.pop(0))


def scholars_mate(**kwargs):
    white = ScriptedBot("White", SCHOLARS_MATE_WHITE)
    black = ScriptedBot("Black", SCHOLARS_MATE_BLACK)
    return white, black, play_game(white, black, **kwargs)


class TestPlayGame:
    def test_checkmate_is_scored_and_reported(self):
        white, black, result = scholars_mate()
        assert (result.result, result.winner, result.termination) == ("1-0", "white", "checkmate")
        assert result.moves == ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
        assert result.total_moves == 7 and chess.Board(result.final_fen).is_checkmate()

    def test_players_are_told_when_the_game_starts(self):
        white, black, _ = scholars_mate()
        assert white.events == [("start", chess.WHITE, "Black")]
        assert black.events[0] == ("start", chess.BLACK, "White")

    def test_an_illegal_move_loses_the_game(self):
        white, black = ScriptedBot("W", ["e2e5"]), ScriptedBot("B", [])
        result = play_game(white, black)
        assert (result.winner, result.termination, result.result) == (
            "black",
            "illegal_move",
            "0-1",
        )
        assert result.moves == []

    def test_an_exception_in_a_bot_loses_the_game(self):
        result = play_game(ScriptedBot("W", SCHOLARS_MATE_WHITE), ScriptedBot("B", [], fail=True))
        assert (result.winner, result.termination, result.result) == (
            "white",
            "player_error",
            "1-0",
        )

    def test_exceeding_the_clock_loses_on_time(self, monkeypatch):
        clock = FakeClock()
        monkeypatch.setattr(play_module, "time", SimpleNamespace(time=clock.time))
        slow = ScriptedBot("W", ["e2e4"], delay=0.2, clock=clock)
        result = play_game(slow, ScriptedBot("B", ["e7e5"]), time_limit=0.05)
        assert (result.winner, result.termination, result.result) == (
            "black",
            "time_forfeit",
            "0-1",
        )

    def test_a_position_that_is_already_drawn_ends_immediately(self):
        result = play_game(
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

    def test_increment_is_added_after_every_move(self):
        _, _, result = scholars_mate(time_limit=30.0, time_increment=2.0)
        assert result.white_time_left > 30.0 and result.black_time_left > 30.0

    def test_games_can_start_from_a_custom_position(self):
        fen = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"
        result = play_game(ScriptedBot("W", ["a1a8"]), ScriptedBot("B", []), starting_position=fen)
        assert result.starting_fen.startswith("6k1/5ppp") and result.termination == "checkmate"

    def test_verbose_mode_prints_the_game(self, capsys):
        scholars_mate(verbose=True)
        output = capsys.readouterr().out
        assert "Starting game" in output and "Qxf7#" in output
