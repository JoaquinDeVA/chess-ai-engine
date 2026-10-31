"""Behaviour of the bot: legality, tactics, time handling and the regressions B1-B16."""

import shutil
import time

import chess
import pytest

from chess_engine.bots.search_bot import HEURISTIC, MAX_DEPTH, SearchBot, fallback_move
from chess_engine.bots.time_manager import TimeManager
from chess_engine.knowledge.opening_book import OpeningBook
from chess_engine.models import load_evaluator, loader
from tests.conftest import MATE_IN_ONE

MATE_MOVE = chess.Move.from_uci("a1a8")


def make_bots(material_model):
    """One instance of every bot that does not need a trained artifact."""
    return {
        "heuristic": SearchBot(HEURISTIC, max_depth=3, opening_book=None, tablebase=None),
        "model_book": SearchBot(material_model, opening_book=OpeningBook({}), max_depth=3),
    }


@pytest.mark.parametrize("name", ["heuristic", "model_book"])
class TestEveryBot:
    def test_returns_a_legal_move_from_several_positions(self, material_model, sample_boards, name):
        bot = make_bots(material_model)[name]
        for board in sample_boards[:6]:
            move = bot.get_move(board, 10.0)
            assert isinstance(move, chess.Move), "Regression B1/B2: get_move must return a Move"
            assert move in board.legal_moves

    def test_finds_mate_in_one(self, material_model, name):
        bot = make_bots(material_model)[name]
        assert bot.get_move(chess.Board(MATE_IN_ONE), 10.0) == MATE_MOVE

    def test_does_not_modify_the_board_it_is_given(self, material_model, name):
        board = chess.Board()
        board.push_san("e4")
        before = board.fen(), len(board.move_stack)
        make_bots(material_model)[name].get_move(board, 10.0)
        assert (board.fen(), len(board.move_stack)) == before

    def test_statistics_are_kept_and_reset_on_a_new_game(self, material_model, name):
        bot = make_bots(material_model)[name]
        bot.get_move(chess.Board(), 10.0)
        assert bot.last_search_stats.nodes_visited > 0
        bot.on_game_start(chess.WHITE, "opponent")
        assert bot.last_search_stats is None


class TestFallback:
    def test_a_search_that_returns_no_move_falls_back_to_a_legal_one(self):
        # no time at all: the hard limit has passed at the root and the search yields no move
        hopeless = SearchBot(HEURISTIC, opening_book=None, tablebase=None)
        hopeless.time_manager = TimeManager(min_turn_time=0.0, safety_margin=1.0)
        board = chess.Board()
        assert hopeless.search(board, 0.0) is None
        assert hopeless.get_move(board, 0.0) in board.legal_moves


class TestSearch:
    def test_the_capture_ordering_is_used(self):
        """Regression B8: the MVV-LVA ordering was defined but never passed to the search."""
        board = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7")
        nodes = {}
        for disabled in ((), ("ordering",)):
            bot = SearchBot(
                HEURISTIC, max_depth=4, quiescence_plies=0, opening_book=None, tablebase=None
            )
            bot.disabled = disabled
            bot.get_move(board, 1e6)
            nodes[disabled] = bot.last_search_stats.nodes_visited
        assert nodes[()] < nodes[("ordering",)]

    def test_stays_within_the_time_budget(self):
        bot = SearchBot(HEURISTIC, max_depth=30, opening_book=None, tablebase=None)
        started = time.time()
        move = bot.get_move(chess.Board(), 2.0)
        assert time.time() - started < 2.0
        assert move in chess.Board().legal_moves

    def test_returns_a_legal_move_when_there_is_almost_no_time(self):
        bot = SearchBot(HEURISTIC, max_depth=5, opening_book=None, tablebase=None)
        board = chess.Board()
        assert bot.get_move(board, 0.05) in board.legal_moves

    def test_deepens_beyond_the_first_iterations_when_time_allows(self):
        bot = SearchBot(HEURISTIC, max_depth=6, opening_book=None, tablebase=None)
        bot.get_move(chess.Board(MATE_IN_ONE.replace("R3K3", "4K3")), 20.0)
        assert bot.last_search_stats.max_depth_reached >= 3

    def test_without_a_depth_limit_the_clock_decides(self):
        """Regression: bots used to stop at depth 3 by default."""
        bot = SearchBot(HEURISTIC, opening_book=None, tablebase=None)
        bot.get_move(chess.Board(), 20.0)
        assert bot.last_search_stats.completed_depth > 3
        assert bot.max_depth == MAX_DEPTH

    def test_several_threads_play_a_legal_move_within_the_budget(self):
        bot = SearchBot(
            HEURISTIC, max_depth=8, threads=3, hash_mb=8, opening_book=None, tablebase=None
        )
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        started = time.time()
        assert bot.get_move(board, 2.0) in board.legal_moves
        assert time.time() - started < 2.0
        assert bot.last_search_stats.nodes_visited > 0 and bot.threads == 3

    def test_an_unknown_technique_is_rejected(self):
        bot = SearchBot(
            HEURISTIC, max_depth=2, techniques=("warp_speed",), opening_book=None, tablebase=None
        )
        with pytest.raises(ValueError, match="unknown technique"):
            bot.search(chess.Board(), 1.0)


class TestModelAndBook:
    def test_evaluator_scale_uses_the_whole_interval(self, material_model):
        """Regression B10: centipawns were divided by 20000, squeezing evaluations into 0.47-0.53."""
        evaluator = load_evaluator(material_model)
        up_a_queen = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        white, black = evaluator.evaluate(up_a_queen, True), evaluator.evaluate(up_a_queen, False)
        assert white > 0.7 and black < 0.3
        assert white + black == pytest.approx(1.0)

    def test_the_bot_prefers_winning_material(self, material_model):
        board = chess.Board("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1")  # Qxd5 wins the queen
        bot = SearchBot(material_model, opening_book=OpeningBook({}), max_depth=2)
        assert bot.get_move(board, 10.0) == chess.Move.from_uci("d1d5")

    def test_opening_book_is_used_until_the_game_leaves_it(self, material_model):
        book = OpeningBook({chess.Board().fen(): "e2e4"})
        bot = SearchBot(material_model, opening_book=book, max_depth=2)
        board = chess.Board()
        assert bot.get_move(board, 10.0) == chess.Move.from_uci("e2e4")
        assert bot.in_book
        board.push_san("e4")
        board.push_san("a6")  # not in the (one-entry) book
        board.push_san("Nf3")
        bot.get_move(board, 10.0)
        assert not bot.in_book
        assert (
            bot.get_move(chess.Board(), 10.0) in chess.Board().legal_moves
        )  # book no longer consulted

    def test_the_book_is_an_option_of_the_bot(self):
        book = OpeningBook({chess.Board().fen(): "d2d4"})
        with_book = SearchBot(HEURISTIC, max_depth=1, opening_book=book)
        assert with_book.get_move(chess.Board(), 10.0) == chess.Move.from_uci("d2d4")
        assert with_book.last_search_stats.stop_reason == "book"
        assert SearchBot(HEURISTIC, max_depth=1, opening_book=None).opening_book is None

    def test_the_shipped_book_and_tables_are_on_unless_switched_off(self):
        bot = SearchBot(HEURISTIC, max_depth=1)
        assert bot.opening_book is OpeningBook.shipped() and len(bot.opening_book) > 0
        assert SearchBot(HEURISTIC, max_depth=1, tablebase=None).tablebase is None

    def test_a_new_game_re_enables_the_opening_book(self, material_model):
        """Regression B16: the book flag was never reset, so the second game had no book."""
        book = OpeningBook({chess.Board().fen(): "e2e4"})
        bot = SearchBot(material_model, opening_book=book, max_depth=2)
        bot.in_book = False
        bot.on_game_start(chess.WHITE, "opponent")
        assert bot.get_move(chess.Board(), 10.0) == chess.Move.from_uci("e2e4")

    def test_models_can_be_selected_by_name(self, tmp_path, monkeypatch):
        shutil.copy(loader.artifact_path("linear_material"), tmp_path / "linear_material.npz")
        monkeypatch.setattr(loader, "ARTIFACTS_DIR", tmp_path)
        bot = SearchBot("linear_material", max_depth=1)
        assert bot.name == "linear_material"

    def test_default_model_loads(self):
        """Regression B12: the default pointed to a file that did not exist in a fresh clone."""
        board = chess.Board()
        assert SearchBot(max_depth=2).get_move(board, 10.0) in board.legal_moves


def test_fallback_move_prefers_captures():
    board = chess.Board("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1")
    assert fallback_move(board) == chess.Move.from_uci("d1d5")
    assert fallback_move(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == chess.Move.null()
