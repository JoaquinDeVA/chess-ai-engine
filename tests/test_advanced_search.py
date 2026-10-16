"""Tests for the advanced alpha-beta search and iterative deepening."""

from collections import defaultdict

import chess

from chess_engine.search.chess_game_state import ChessGameState
from chess_engine.search.chess_problem import ChessProblem
from chess_engine.search.game_algorithms import (
    SearchStatistics,
    advanced_heuristic_alphabeta_search,
    heuristic_alphabeta_iterative_deepening_search,
)
from tests.helpers import TreeGame, TreeState, never_cutoff, zero_eval


def build_state(tree, utilities):
    return TreeState("root", tree, utilities)


def test_returns_move_and_statistics():
    tree = {"root": {"a": "x", "b": "y"}}
    utilities = {"x": 1.0, "y": 0.0}
    move, stats = advanced_heuristic_alphabeta_search(
        TreeGame(), build_state(tree, utilities), zero_eval, never_cutoff, max_depth=2
    )
    assert move == "a"
    assert isinstance(stats, SearchStatistics)
    assert stats.nodes_visited >= 3


def test_prefers_the_faster_win_regardless_of_who_reaches_the_terminal_node():
    """Regression B7: the depth bonus was inverted at minimizing nodes.

    ``slow`` wins after the opponent's reply (ply 2), ``fast`` wins immediately (ply 1).
    The slow move is listed first, so a tie would pick it.
    """
    tree = {
        "root": {"slow": "s1", "fast": "f1"},
        "s1": {"reply": "s2"},
    }
    utilities = {"s2": 1.0, "f1": 1.0}
    move, _ = advanced_heuristic_alphabeta_search(
        TreeGame(), build_state(tree, utilities), zero_eval, never_cutoff, max_depth=3
    )
    assert move == "fast"


def test_prefers_the_slower_loss():
    tree = {
        "root": {"quick_loss": "q1", "slow_loss": "s1"},
        "s1": {"reply": "s2"},
    }
    utilities = {"q1": 0.0, "s2": 0.0}
    move, _ = advanced_heuristic_alphabeta_search(
        TreeGame(), build_state(tree, utilities), zero_eval, never_cutoff, max_depth=3
    )
    assert move == "slow_loss"


def test_killer_moves_are_recorded_and_bounded():
    tree = {"root": {"a": "x", "b": "y"}, "x": {"c": "p", "d": "q"}, "y": {"c": "r", "d": "t"}}
    utilities = {"p": 1.0, "q": 0.0, "r": 1.0, "t": 0.0}
    killers = defaultdict(list)
    advanced_heuristic_alphabeta_search(
        TreeGame(),
        build_state(tree, utilities),
        zero_eval,
        never_cutoff,
        killer_moves=killers,
        max_depth=3,
    )
    assert sum(len(v) for v in killers.values()) > 0
    assert all(len(v) <= 3 for v in killers.values())


def test_transposition_table_is_filled_and_reused():
    tree = {"root": {"a": "x", "b": "y"}}
    utilities = {"x": 1.0, "y": 0.0}
    table = {}
    advanced_heuristic_alphabeta_search(
        TreeGame(),
        build_state(tree, utilities),
        zero_eval,
        never_cutoff,
        transposition_table=table,
        max_depth=2,
    )
    assert len(table) > 0
    _, stats = advanced_heuristic_alphabeta_search(
        TreeGame(),
        build_state(tree, utilities),
        zero_eval,
        never_cutoff,
        transposition_table=table,
        max_depth=2,
    )
    assert stats.nodes_visited <= 2, "second search must be answered from the table"


def test_order_callback_is_invoked():
    tree = {"root": {"a": "x", "b": "y"}}
    utilities = {"x": 0.5, "y": 0.5}
    calls = []

    def reverse(board, moves):
        calls.append(moves)
        return list(reversed(moves))

    move, _ = advanced_heuristic_alphabeta_search(
        TreeGame(),
        build_state(tree, utilities),
        zero_eval,
        never_cutoff,
        order_moves_callback=reverse,
        max_depth=2,
    )
    assert calls and move == "b", "with equal values the first move tried is kept"


def test_time_truncated_searches_do_not_pollute_the_transposition_table():
    tree = {"root": {"a": "x", "b": "y"}, "x": {"c": "p"}, "y": {"c": "q"}}
    utilities = {"p": 1.0, "q": 0.0}
    table = {}

    def expire_below_root(state, depth_remaining, elapsed):
        return depth_remaining < 3

    _, stats = advanced_heuristic_alphabeta_search(
        TreeGame(),
        build_state(tree, utilities),
        zero_eval,
        expire_below_root,
        transposition_table=table,
        max_depth=3,
    )
    assert stats.truncated
    assert table == {}


class TestIterativeDeepening:
    def test_returns_the_deepest_completed_result(self):
        tree = {"root": {"a": "x", "b": "y"}, "x": {"c": "p"}, "y": {"c": "q"}}
        utilities = {"p": 1.0, "q": 0.0}
        move, stats = heuristic_alphabeta_iterative_deepening_search(
            TreeGame(),
            build_state(tree, utilities),
            zero_eval,
            never_cutoff,
            max_depth=4,
            time_limit=5.0,
        )
        assert move == "a"
        assert isinstance(stats, SearchStatistics)

    def test_discards_an_iteration_interrupted_by_the_clock(self):
        """Regression B6: an unfinished iteration must not override a completed one.

        The simulated clock expires on the 9th cut-off check, i.e. right at the root of the
        third iteration, which therefore returns no move at all.
        """
        tree = {
            "root": {"good": "g1", "bad": "b1"},
            "g1": {"r": "g2"},
            "b1": {"r": "b2"},
            "g2": {"r": "g3"},
            "b2": {"r": "b3"},
        }
        utilities = {"g3": 1.0, "b3": 0.0}
        calls = {"n": 0}

        def eval_fn(state, player):
            return 0.8 if state.name.startswith("g") else 0.2

        def clock_runs_out_in_depth_three(state, depth_remaining, elapsed):
            calls["n"] += 1
            return depth_remaining == 0 or calls["n"] > 8

        move, stats = heuristic_alphabeta_iterative_deepening_search(
            TreeGame(),
            build_state(tree, utilities),
            eval_fn,
            clock_runs_out_in_depth_three,
            max_depth=3,
            time_limit=60.0,
        )
        assert calls["n"] > 8, "the third iteration must have started"
        assert move == "good"
        assert not stats.truncated

    def test_stops_when_the_next_iteration_would_not_fit(self):
        tree = {"root": {"a": "x"}, "x": {"c": "p"}}
        utilities = {"p": 1.0}
        move, _ = heuristic_alphabeta_iterative_deepening_search(
            TreeGame(),
            build_state(tree, utilities),
            zero_eval,
            never_cutoff,
            max_depth=50,
            time_limit=0.2,
        )
        assert move == "a"


def test_tree_state_is_restored_after_the_search_also_on_cutoffs():
    tree = {"root": {"a": "x", "b": "y"}, "x": {"c": "p", "d": "q"}, "y": {"c": "p", "d": "q"}}
    utilities = {"p": 1.0, "q": 0.0}
    for cutoff in (never_cutoff, lambda state, depth, elapsed: depth == 1):
        state = build_state(tree, utilities)
        advanced_heuristic_alphabeta_search(TreeGame(), state, zero_eval, cutoff, max_depth=3)
        assert (state.name, state.to_move, state._undo_stack) == ("root", 0, [])


def test_state_is_restored_when_the_search_raises():
    tree = {"root": {"a": "x"}, "x": {"c": "p"}}
    state = build_state(tree, {"p": 1.0})

    def failing_eval(node, player):
        raise RuntimeError("boom")

    try:
        advanced_heuristic_alphabeta_search(
            TreeGame(), state, failing_eval, never_cutoff, max_depth=1
        )
    except RuntimeError:
        pass
    assert (state.name, state.to_move) == ("root", 0)


def test_chess_root_state_is_exactly_restored_by_iterative_deepening():
    board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    for uci in ["a1b1", "a8b8", "b1a1", "b8a8"]:
        board.push_uci(uci)
    state = ChessGameState(board)
    state.is_over()
    before = (
        state.board.fen(),
        list(state.board.move_stack),
        state._history,
        state._has_repeated,
        state._outcome,
        state._outcome_known,
    )

    def material(node, player):
        return 0.5 + 0.01 * len(node.board.piece_map()) * (1 if player else -1)

    move, _ = heuristic_alphabeta_iterative_deepening_search(
        ChessProblem(board), state, material, never_cutoff, max_depth=3, time_limit=60.0
    )
    assert move in board.legal_moves
    after = (
        state.board.fen(),
        list(state.board.move_stack),
        state._history,
        state._has_repeated,
        state._outcome,
        state._outcome_known,
    )
    assert after == before
    assert state._undo_stack == []
