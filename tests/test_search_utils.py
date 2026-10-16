"""Tests for the helpers shared by the advanced search (transposition table, killers, ordering)."""

import chess

from adversarial_search.game_algorithms import AdversarialSearchResult
from adversarial_search.search_utils import (
    EXACT,
    LOWER_BOUND,
    UPPER_BOUND,
    TTEntry,
    mvv_lva_score,
    order_by_mvv_lva,
    probe_transposition_table,
    record_killer_move,
    reorder_by_killer_moves,
    store_transposition_entry,
)


def result(value, move="m"):
    return AdversarialSearchResult(value, move)


class TestTranspositionTable:
    def test_exact_entry_with_enough_depth_is_returned(self):
        table = {"k": TTEntry(result(0.7), depth_remaining=3, flag=EXACT)}
        hit, alpha, beta = probe_transposition_table(table, "k", 2, -1.0, 1.0)
        assert hit.value == 0.7
        assert (alpha, beta) == (-1.0, 1.0)

    def test_shallow_entry_is_ignored(self):
        table = {"k": TTEntry(result(0.7), depth_remaining=1, flag=EXACT)}
        hit, alpha, beta = probe_transposition_table(table, "k", 2, -1.0, 1.0)
        assert hit is None and (alpha, beta) == (-1.0, 1.0)

    def test_lower_bound_raises_alpha_and_can_cut(self):
        table = {"k": TTEntry(result(0.6), depth_remaining=2, flag=LOWER_BOUND)}
        hit, alpha, _ = probe_transposition_table(table, "k", 2, 0.0, 1.0)
        assert hit is None and alpha == 0.6
        hit, _, _ = probe_transposition_table(table, "k", 2, 0.0, 0.5)
        assert hit is not None

    def test_upper_bound_lowers_beta(self):
        table = {"k": TTEntry(result(0.4), depth_remaining=2, flag=UPPER_BOUND)}
        hit, _, beta = probe_transposition_table(table, "k", 2, 0.0, 1.0)
        assert hit is None and beta == 0.4

    def test_flag_depends_on_original_window(self):
        table = {}
        store_transposition_entry(table, "a", result(0.0), 1, alpha_orig=0.1, beta_orig=0.9)
        store_transposition_entry(table, "b", result(1.0), 1, alpha_orig=0.1, beta_orig=0.9)
        store_transposition_entry(table, "c", result(0.5), 1, alpha_orig=0.1, beta_orig=0.9)
        assert table["a"].flag == UPPER_BOUND
        assert table["b"].flag == LOWER_BOUND
        assert table["c"].flag == EXACT

    def test_table_size_is_bounded(self):
        table = {}
        for i in range(25):
            store_transposition_entry(table, i, result(0.5), 1, 0.0, 1.0, max_entries=10)
        assert len(table) <= 10
        assert 24 in table, "the newest entry must survive eviction"


class TestKillerMoves:
    def test_at_most_three_killers_per_depth_and_no_duplicates(self):
        killers = {}
        for move in ["a", "b", "c", "c", "d"]:
            record_killer_move(killers, 2, move)
        assert killers[2] == ["b", "c", "d"]

    def test_killers_are_tried_first_without_losing_moves(self):
        killers = {3: ["x"]}
        assert reorder_by_killer_moves(["a", "x", "b"], 3, killers) == ["x", "a", "b"]


class TestMvvLva:
    def test_capture_of_queen_by_pawn_beats_capture_of_pawn_by_queen(self):
        board = chess.Board("4k3/8/8/3q4/4P3/8/8/3QK3 w - - 0 1")
        pawn_takes_queen = chess.Move.from_uci("e4d5")
        queen_takes_queen = chess.Move.from_uci("d1d5")
        assert mvv_lva_score(board, pawn_takes_queen) > 0
        assert mvv_lva_score(board, pawn_takes_queen) > mvv_lva_score(
            board, chess.Move.from_uci("d1d2")
        )
        assert mvv_lva_score(board, pawn_takes_queen) > mvv_lva_score(board, queen_takes_queen)

    def test_en_passant_counts_as_a_pawn_capture(self):
        """Regression B9: the victim square of an en-passant capture is empty."""
        board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        move = chess.Move.from_uci("e5d6")
        assert board.is_en_passant(move)
        assert mvv_lva_score(board, move) > 0

    def test_captures_come_first_and_all_moves_are_kept(self):
        board = chess.Board("4k3/8/8/3q4/4P3/8/8/3QK3 w - - 0 1")
        moves = list(board.legal_moves)
        ordered = order_by_mvv_lva(board, moves)
        assert sorted(map(str, ordered)) == sorted(map(str, moves))
        assert board.is_capture(ordered[0])
