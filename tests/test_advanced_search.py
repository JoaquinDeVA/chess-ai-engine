"""Tests for the advanced alpha-beta search and iterative deepening."""

import types
from collections import defaultdict

import chess

from chess_engine.search import game_algorithms
from chess_engine.search.chess_game_state import ChessGameState
from chess_engine.search.chess_problem import ChessProblem, chess_noisy_moves
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

    def test_completed_depth_excludes_truncated_iteration(self):
        """The cut-off fires inside the depth-3 iteration after it already reached ply 3."""
        tree = {
            "root": {"good": "g1", "bad": "y1"},
            "g1": {"r": "g2"},
            "g2": {"r": "g3"},
            "g3": {"r": "g4"},
            "y1": {"r": "y2"},
        }

        def cutoff_at_y1_in_depth_three(state, depth_remaining, elapsed):
            return state.name == "y1" and depth_remaining == 2

        _, stats = heuristic_alphabeta_iterative_deepening_search(
            TreeGame(),
            build_state(tree, {"y2": 0.0, "g4": 0.0}),
            zero_eval,
            cutoff_at_y1_in_depth_three,
            max_depth=3,
            time_limit=60.0,
        )
        assert stats.completed_depth == 2
        assert stats.max_depth_reached == 3

    def test_completed_depth_equals_max_depth_without_cutoff(self):
        tree = {
            "root": {"a": "x1"},
            "x1": {"r": "x2"},
            "x2": {"r": "x3"},
            "x3": {"r": "x4"},
        }
        _, stats = heuristic_alphabeta_iterative_deepening_search(
            TreeGame(),
            build_state(tree, {"x4": 0.0}),
            zero_eval,
            never_cutoff,
            max_depth=3,
            time_limit=60.0,
        )
        assert stats.completed_depth == 3
        assert stats.max_depth_reached == 3

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


PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material_eval(state, player):
    """Material balance for ``player`` squashed to [0, 1]."""
    board = state.get_board()
    balance = sum(
        (1 if piece.color == player else -1) * PIECE_VALUES.get(piece.piece_type, 0)
        for piece in board.piece_map().values()
    )
    return 0.5 + balance / 80


def _chess_search(fen, depth, **kwargs):
    return advanced_heuristic_alphabeta_search(
        ChessProblem(),
        ChessGameState(chess.Board(fen)),
        material_eval,
        lambda node, depth_left, elapsed: depth_left == 0,
        max_depth=depth,
        **kwargs,
    )


QUIESCENCE_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1",
]


def test_quiescence_off_by_default_keeps_move_and_nodes():
    for fen in QUIESCENCE_FENS:
        baseline = _chess_search(fen, 3)
        explicit = _chess_search(fen, 3, noisy_moves=chess_noisy_moves, quiescence_plies=0)
        without_function = _chess_search(fen, 3, quiescence_plies=4)
        assert explicit[0] == baseline[0] and explicit[1] == baseline[1]
        assert without_function[0] == baseline[0] and without_function[1] == baseline[1]


def test_quiescence_sees_the_recapture_at_the_horizon():
    """Qxd5 wins a pawn on paper but cxd5 follows: only quiescence notices."""
    fen = "4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1"
    blind, _ = _chess_search(fen, 1)
    careful, stats = _chess_search(fen, 1, noisy_moves=chess_noisy_moves, quiescence_plies=2)
    assert blind == chess.Move.from_uci("d1d5")
    assert careful != chess.Move.from_uci("d1d5")
    assert stats.nodes_visited > _chess_search(fen, 1)[1].nodes_visited


def test_quiescence_stand_pat_without_captures():
    """With no noisy moves the value is the static one, never worse for the side to move."""
    fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    board = chess.Board(fen)
    assert chess_noisy_moves(ChessGameState(board)) == []
    quiet = _chess_search(fen, 2)
    with_q = _chess_search(fen, 2, noisy_moves=chess_noisy_moves, quiescence_plies=3)
    assert with_q[0] == quiet[0]
    assert with_q[1].nodes_visited == quiet[1].nodes_visited


def test_chess_noisy_moves_are_captures_and_promotions():
    board = chess.Board("1n2k3/P7/8/8/8/8/8/R3K3 w - - 0 1")
    moves = {m.uci() for m in chess_noisy_moves(ChessGameState(board))}
    assert moves == {"a7a8q", "a7a8r", "a7a8b", "a7a8n", "a7b8q", "a7b8r", "a7b8b", "a7b8n"}


def test_iterative_deepening_passes_the_table_cap_to_the_search():
    table = {}
    heuristic_alphabeta_iterative_deepening_search(
        ChessProblem(),
        ChessGameState(chess.Board()),
        material_eval,
        never_cutoff,
        transposition_table=table,
        max_depth=3,
        time_limit=60.0,
        max_tt_entries=50,
    )
    assert 0 < len(table) <= 50


def _fake_clock_search(monkeypatch, durations):
    """Replace the single-depth search by one that only advances a fake clock by ``durations``."""
    clock = [0.0]

    def fake_search(*, max_depth, **kwargs):
        clock[0] += durations[max_depth]
        return "m", SearchStatistics(1, 0, max_depth, False, max_depth)

    monkeypatch.setattr(game_algorithms, "advanced_heuristic_alphabeta_search", fake_search)
    monkeypatch.setattr(game_algorithms, "time", types.SimpleNamespace(time=lambda: clock[0]))


def test_time_estimate_cannot_stop_the_deepening_before_min_depth(monkeypatch):
    # depth 2 is 100 times slower than depth 1, so the extrapolation predicts 10 s for depth 3
    _fake_clock_search(monkeypatch, {1: 0.001, 2: 0.1, 3: 1.0, 4: 10.0})
    _, stats = heuristic_alphabeta_iterative_deepening_search(
        None, None, None, None, max_depth=4, time_limit=5.0, min_depth=3
    )
    assert stats.completed_depth == 3


def test_time_estimate_still_stops_the_deepening_after_min_depth(monkeypatch):
    _fake_clock_search(monkeypatch, {1: 0.001, 2: 0.1, 3: 1.0, 4: 10.0})
    _, stats = heuristic_alphabeta_iterative_deepening_search(
        None, None, None, None, max_depth=4, time_limit=5.0, min_depth=1
    )
    assert stats.completed_depth == 2


def test_exhausted_soft_budget_still_completes_min_depth():
    _, stats = heuristic_alphabeta_iterative_deepening_search(
        ChessProblem(),
        ChessGameState(chess.Board()),
        zero_eval,
        never_cutoff,
        max_depth=5,
        time_limit=0.0,
        min_depth=3,
    )
    assert stats.completed_depth == 3
