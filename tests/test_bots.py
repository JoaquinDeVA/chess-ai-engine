"""Behaviour of every bot: legality, tactics, time handling and the regressions B1-B16."""

import time

import chess
import pytest

from bots._search_bots import IterativeDeepeningBot, fallback_move
from bots.iterative_heuristic_bot import IterativeHeuristicBot
from bots.iterative_model_book_bot import IterativeModelBookBot
from evaluation.heuristics import h1, h2, h3, h4
from knowledge.opening_book import OpeningBook
from models.base import ARTIFACTS_DIR
from models.evaluator import ModelEvaluator
from tests.conftest import MATE_IN_ONE

MATE_MOVE = chess.Move.from_uci("a1a8")


def make_bots(stub_model):
    """One instance of every bot that does not need Prolog or a trained artifact."""
    return {
        "heuristic": IterativeHeuristicBot(max_depth=3),
        "model_book": IterativeModelBookBot(
            model=stub_model, opening_book=OpeningBook({}), max_depth=3
        ),
    }


@pytest.mark.parametrize("name", ["heuristic", "model_book"])
class TestEveryBot:
    def test_returns_a_legal_move_from_several_positions(self, stub_model, sample_boards, name):
        bot = make_bots(stub_model)[name]
        for board in sample_boards[:6]:
            move = bot.get_move(board, 10.0)
            assert isinstance(move, chess.Move), "Regression B1/B2: get_move must return a Move"
            assert move in board.legal_moves

    def test_finds_mate_in_one(self, stub_model, name):
        bot = make_bots(stub_model)[name]
        assert bot.get_move(chess.Board(MATE_IN_ONE), 10.0) == MATE_MOVE

    def test_does_not_modify_the_board_it_is_given(self, stub_model, name):
        board = chess.Board()
        board.push_san("e4")
        before = board.fen(), len(board.move_stack)
        make_bots(stub_model)[name].get_move(board, 10.0)
        assert (board.fen(), len(board.move_stack)) == before

    def test_statistics_are_collected_and_reset_on_a_new_game(self, stub_model, name):
        bot = make_bots(stub_model)[name]
        bot.get_move(chess.Board(), 10.0)
        assert bot.get_search_statistics()["move_count"] == 1
        bot.on_game_start(chess.WHITE, "opponent")
        assert bot.get_search_statistics()["move_count"] == 0

    def test_info_describes_the_bot(self, stub_model, name):
        info = make_bots(stub_model)[name].get_info()
        assert info["author"] and info["description"]


class TestEvaluation:
    def test_all_heuristics_stay_inside_the_open_unit_interval(self, sample_boards):
        from adversarial_search.chess_game_state import ChessGameState

        for heuristic in (h1, h2, h3, h4):
            for board in sample_boards[:15]:
                for player in (chess.WHITE, chess.BLACK):
                    assert 0.0 < heuristic(ChessGameState(board), player) < 1.0

    def test_a_search_that_returns_no_move_falls_back_to_a_legal_one(self, stub_model):
        class Hopeless(IterativeDeepeningBot):
            def __init__(self):
                super().__init__("Hopeless")

            def evaluate(self, state, player):
                return 0.5

            def cutoff_test(self, state, depth_left, time_used):
                return True  # cut off at the root: the search yields no move

        board = chess.Board()
        assert Hopeless().get_move(board, 1.0) in board.legal_moves


class TestIterativeDeepening:
    def test_transposition_table_and_killers_fill_up_and_clear_on_new_game(self):
        bot = IterativeHeuristicBot(max_depth=3)
        bot.get_move(chess.Board(), 10.0)
        assert bot.transposition_table and bot.killer_moves
        bot.on_game_start(chess.BLACK, "opponent")
        assert not bot.transposition_table and not bot.killer_moves

    def test_the_capture_ordering_callback_is_used(self):
        """Regression B8: the MVV-LVA callback was defined but never passed to the search."""
        bot = IterativeHeuristicBot(max_depth=2)
        calls = []
        original = bot.order_moves
        bot.order_moves = lambda board, moves: calls.append(len(moves)) or original(board, moves)
        bot.get_move(chess.Board(), 10.0)
        assert calls

    def test_stays_within_the_time_budget(self):
        bot = IterativeHeuristicBot(max_depth=30)
        started = time.time()
        move = bot.get_move(chess.Board(), 2.0)
        assert time.time() - started < 2.0
        assert move in chess.Board().legal_moves

    def test_returns_a_legal_move_when_there_is_almost_no_time(self):
        bot = IterativeHeuristicBot(max_depth=5)
        board = chess.Board()
        assert bot.get_move(board, 0.05) in board.legal_moves

    def test_deepens_beyond_the_first_iterations_when_time_allows(self):
        bot = IterativeHeuristicBot(max_depth=6)
        bot.get_move(chess.Board(MATE_IN_ONE.replace("R3K3", "4K3")), 20.0)
        assert bot.last_search_stats.max_depth_reached >= 3

    def test_abstract_bases_are_not_playable(self):
        with pytest.raises(TypeError):
            IterativeDeepeningBot("x")


class TestMLBots:
    def test_evaluator_scale_uses_the_whole_interval(self, stub_model):
        """Regression B10: centipawns were divided by 20000, squeezing evaluations into 0.47-0.53."""
        from adversarial_search.chess_game_state import ChessGameState

        evaluate = ModelEvaluator(stub_model)
        up_a_queen = ChessGameState(
            chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        )
        assert evaluate(up_a_queen, chess.WHITE) > 0.7
        assert evaluate(up_a_queen, chess.BLACK) < 0.3
        assert evaluate(up_a_queen, chess.WHITE) + evaluate(
            up_a_queen, chess.BLACK
        ) == pytest.approx(1.0)

    def test_the_bot_prefers_winning_material(self, stub_model):
        board = chess.Board("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1")  # Qxd5 wins the queen
        bot = IterativeModelBookBot(model=stub_model, opening_book=OpeningBook({}), max_depth=2)
        assert bot.get_move(board, 10.0) == chess.Move.from_uci("d1d5")

    def test_opening_book_is_used_until_the_game_leaves_it(self, stub_model):
        book = OpeningBook({chess.Board().fen(): "e2e4"})
        bot = IterativeModelBookBot(model=stub_model, opening_book=book, max_depth=2)
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

    def test_a_new_game_re_enables_the_opening_book(self, stub_model):
        """Regression B16: the book flag was never reset, so the second game had no book."""
        book = OpeningBook({chess.Board().fen(): "e2e4"})
        bot = IterativeModelBookBot(model=stub_model, opening_book=book, max_depth=2)
        bot.in_book = False
        bot.on_game_start(chess.WHITE, "opponent")
        assert bot.get_move(chess.Board(), 10.0) == chess.Move.from_uci("e2e4")

    def test_models_can_be_selected_by_name(self):
        bot = IterativeModelBookBot(model="linear_baseline", max_depth=1)
        assert bot.get_info()["model"] == "linear_baseline"

    @pytest.mark.skipif(
        not (ARTIFACTS_DIR / "mlp_features.keras").exists(), reason="mlp_features not trained"
    )
    def test_default_model_loads(self):
        """Regression B12: the default pointed to a file that did not exist."""
        pytest.importorskip("keras")
        board = chess.Board()
        assert IterativeModelBookBot(max_depth=2).get_move(board, 10.0) in board.legal_moves


def test_fallback_move_prefers_captures():
    board = chess.Board("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1")
    assert fallback_move(board) == chess.Move.from_uci("d1d5")
    assert fallback_move(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == chess.Move.null()
