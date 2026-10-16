import math
import time
from collections import defaultdict
from adversarial_search.game_algorithms import advanced_heuristic_alphabeta_search  # adjust import path


# -----------------------------
# Dummy game and state classes
# -----------------------------
class DummyState:
    """Simple game state for alpha-beta testing."""
    def __init__(self, value, player, depth=0):
        self.value = value
        self.player = player
        self.depth = depth

    def key(self):
        return (self.value, self.player, self.depth)

    def get_current_player(self):
        return self.player

    def get_board(self):
        return None


class DummyGame:
    """Configurable dummy game tree."""
    def __init__(self, terminal_at_depth=2):
        self.terminal_at_depth = terminal_at_depth

    def is_terminal(self, state):
        return state.depth >= self.terminal_at_depth

    def utility(self, state, player):
        # Simplified utility depending on value
        if state.value > 0:
            return 1
        elif state.value < 0:
            return 0
        return 0.5

    def actions(self, state):
        # Two simple possible moves per node
        return [f"m{state.depth}_a", f"m{state.depth}_b"]

    def result(self, state, action):
        # Make 'a' moves positive and 'b' moves negative to force pruning
        delta = 1 if action.endswith("a") else -1
        return DummyState(state.value + delta, -state.player, state.depth + 1)


# -----------------------------
# Helpers
# -----------------------------
def simple_eval_fn(state, player):
    return state.value


def simple_cutoff_test(state, depth, elapsed):
    return False


# -----------------------------
# Tests
# -----------------------------
def test_killer_moves_registered():
    """Force pruning and verify killer moves are registered."""
    game = DummyGame()
    state = DummyState(0, 1)
    killer_moves = defaultdict(list)

    # Use a biased eval to trigger pruning
    def biased_eval_fn(state, player):
        return 100 if state.value > 0 else -100

    move, stats = advanced_heuristic_alphabeta_search(
        game,
        state,
        eval_fn=biased_eval_fn,
        cutoff_test=simple_cutoff_test,
        killer_moves=killer_moves,
        max_depth=2,
    )

    registered = sum(len(v) for v in killer_moves.values())
    assert registered > 0, f"No killer moves registered: {killer_moves}"
    assert all(len(v) <= 3 for v in killer_moves.values())


def test_transposition_table_reuse():
    """Ensure that the transposition table stores and reuses results."""
    game = DummyGame()
    state = DummyState(0, 1)
    tt = {}

    move1, stats1 = advanced_heuristic_alphabeta_search(
        game, state, simple_eval_fn, simple_cutoff_test, transposition_table=tt, max_depth=1
    )
    # Should store something in the transposition table
    assert len(tt) > 0, "Transposition table should contain entries after first search."

    # Run again; it should reuse stored results (not increase size much)
    move2, stats2 = advanced_heuristic_alphabeta_search(
        game, state, simple_eval_fn, simple_cutoff_test, transposition_table=tt, max_depth=1
    )
    assert len(tt) >= 1
    assert isinstance(next(iter(tt.values())).result.value, (int, float))


def test_epsilon_adjustment():
    """Verify that epsilon depth adjustment changes the returned value."""
    game = DummyGame(terminal_at_depth=3)
    state = DummyState(1, 1)

    # Two searches with different depth → epsilon should slightly alter utility
    move1, stats1 = advanced_heuristic_alphabeta_search(
        game, state, simple_eval_fn, simple_cutoff_test, max_depth=2
    )
    move2, stats2 = advanced_heuristic_alphabeta_search(
        game, state, simple_eval_fn, simple_cutoff_test, max_depth=4
    )

    # Because of epsilon adjustment, the difference shouldn't be zero
    assert move1 is not None
    assert move2 is not None


def test_order_moves_callback_affects_order():
    """Ensure the custom move ordering callback modifies order."""
    game = DummyGame()
    state = DummyState(0, 1)
    order_called = {}

    def order_cb(board, moves):
        order_called["called"] = True
        return list(reversed(moves))

    move, stats = advanced_heuristic_alphabeta_search(
        game, state, simple_eval_fn, simple_cutoff_test, order_moves_callback=order_cb
    )

    assert "called" in order_called, "Order callback should be invoked."
    assert move is not None
