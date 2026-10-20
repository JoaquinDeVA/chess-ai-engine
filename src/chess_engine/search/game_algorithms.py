"""Adversarial search algorithms.

Contents, from the simplest to the most elaborate:

* :func:`heuristic_alphabeta_search` - alpha-beta with a heuristic evaluation and a cut-off test.
* :func:`advanced_heuristic_alphabeta_search` - adds a transposition table, killer moves, move
  ordering and a depth-aware preference for faster wins.
* :func:`heuristic_alphabeta_iterative_deepening_search` - anytime wrapper over the advanced search.

The three heuristic searches share one return contract: ``(best_move, SearchStatistics)``.

The base classes of the course framework (``GameProblem``/``GameState``) are the work of the
*Inteligencia Artificial e Ingeniería del Conocimiento* course (CEU San Pablo University);
the algorithms below are original work.
"""

import time
from typing import Any, Callable, Dict, Hashable, List, NamedTuple, Optional, Tuple

import chess

from .game_problems import GameProblem, GameState
from .search_utils import (
    DEFAULT_MAX_TT_ENTRIES,
    TTEntry,
    probe_transposition_table,
    record_killer_move,
    reorder_by_killer_moves,
    store_transposition_entry,
)


class AdversarialSearchResult(NamedTuple):
    """
    Result of adversarial search (alpha-beta) containing utility value and best move.

    :param value: The utility value of the position for the searching player
    :type value: float
    :param move: The best move to make, or None if terminal state
    :type move: Any | None
    """

    value: float
    move: Any | None


class SearchStatistics(NamedTuple):
    """
    Statistics collected during alpha-beta search.

    :param nodes_visited: Total number of nodes evaluated
    :type nodes_visited: int
    :param pruning_count: Number of times alpha-beta pruning occurred
    :type pruning_count: int
    :param max_depth_reached: Maximum depth reached in search
    :type max_depth_reached: int
    :param truncated: True if the search was stopped early by the cut-off test (for example
        because the clock ran out) so its result may be based on a shallower search than asked
    :type truncated: bool
    :param completed_depth: Depth of the deepest iterative-deepening iteration that finished
        without being truncated (0 if none did); ``max_depth_reached`` also counts plies seen
        in a truncated iteration
    :type completed_depth: int
    """

    nodes_visited: int
    pruning_count: int
    max_depth_reached: int
    truncated: bool = False
    completed_depth: int = 0


def heuristic_alphabeta_search(
    game: GameProblem,
    state: GameState,
    eval_fn: Callable[[GameState, Any], float],
    cutoff_test: Callable[[GameState, int, float], bool],
    order_moves_callback: Callable[[chess.Board, list[chess.Move]], list[chess.Move]] | None = None,
) -> tuple[Any, SearchStatistics]:
    """
    Search game tree using alpha-beta pruning with heuristic evaluation and cutoff test.

    This is an enhanced version of alphabeta_search that uses:
    - A heuristic evaluation function for non-terminal states
    - A cutoff test to limit search depth based on depth and time

    :param game: The game problem instance containing rules and utility function
    :type game: GameProblem
    :param state: The current game state to analyze
    :type state: GameState
    :param eval_fn: Heuristic evaluation function (state, player) -> float
    :type eval_fn: Callable[[GameState, Any], float]
    :param cutoff_test: Cutoff test function (state, depth, elapsed_time) -> bool
    :type cutoff_test: Callable[[GameState, int, float], bool]
    :param order_moves_callback: A callback that allows the moves to be ordered
    :type order_moves_callback: Callable[[chess.board, list[chess.Move]], list[chess.Move]] | None
    :return: The best move for the current player
    :rtype: Any
    """
    player = state.get_current_player()
    nodes_visited = 0
    pruning_count = 0
    max_depth_reached = 0

    def max_value(
        state: "GameState", alpha: float, beta: float, depth: int
    ) -> AdversarialSearchResult:
        """
        Calculate the maximum value achievable from this state with alpha-beta pruning.

        :param state: The game state to evaluate
        :type state: GameState
        :param alpha: The best value the maximizing player can guarantee so far
        :type alpha: float
        :param beta: The best value the minimizing player can guarantee so far
        :type beta: float
        :param depth: Current search depth (root = 0).
        :type depth: int
        :return: The maximum value and corresponding best move
        :rtype: AdversarialSearchResult
        """
        nonlocal nodes_visited
        nonlocal pruning_count
        nonlocal max_depth_reached

        nodes_visited += 1
        max_depth_reached = max(max_depth_reached, depth)

        if game.is_terminal(state):
            return AdversarialSearchResult(game.utility(state, player), None)

        if cutoff_test(state, depth, time.time() - time_of_start):
            return AdversarialSearchResult(eval_fn(state, player), None)

        best_value = float("-inf")
        best_move = None

        actions = game.actions(state)

        if order_moves_callback:
            actions = order_moves_callback(state.get_board(), actions)

        for action in actions:
            result = min_value(game.result(state, action), alpha, beta, depth + 1)
            if result.value > best_value:
                best_value = result.value
                best_move = action
                alpha = max(alpha, best_value)

            if alpha >= beta:
                pruning_count += 1
                return AdversarialSearchResult(best_value, best_move)

        return AdversarialSearchResult(best_value, best_move)

    def min_value(
        state: "GameState", alpha: float, beta: float, depth: int
    ) -> AdversarialSearchResult:
        """
        Calculate the minimum value achievable from this state with alpha-beta pruning.

        :param state: The game state to evaluate
        :type state: GameState
        :param alpha: The best value the maximizing player can guarantee so far
        :type alpha: float
        :param beta: The best value the minimizing player can guarantee so far
        :type beta: float
        :param depth: Current search depth (root = 0).
        :type depth: int
        :return: The minimum value and corresponding best move
        :rtype: AdversarialSearchResult
        """
        nonlocal nodes_visited
        nonlocal pruning_count
        nonlocal max_depth_reached

        nodes_visited += 1
        max_depth_reached = max(max_depth_reached, depth)

        if game.is_terminal(state):
            return AdversarialSearchResult(game.utility(state, player), None)

        if cutoff_test(state, depth, time.time() - time_of_start):
            return AdversarialSearchResult(eval_fn(state, player), None)

        best_value = float("inf")
        best_move = None

        actions = game.actions(state)

        if order_moves_callback:
            actions = order_moves_callback(state.get_board(), actions)

        for action in actions:
            result = max_value(game.result(state, action), alpha, beta, depth + 1)
            if result.value < best_value:
                best_value = result.value
                best_move = action
                beta = min(beta, best_value)

            if alpha >= beta:
                pruning_count += 1
                return AdversarialSearchResult(best_value, best_move)

        return AdversarialSearchResult(best_value, best_move)

    time_of_start = time.time()
    search_result = max_value(state, float("-inf"), float("inf"), depth=0)
    return search_result.move, SearchStatistics(nodes_visited, pruning_count, max_depth_reached)


#: Weight of the "win faster / lose slower" preference, in utility units per ply.
DEPTH_PREFERENCE_EPSILON = 0.003


def _terminal_utility(
    game: GameProblem, state: GameState, root_player: Any, ply: int, epsilon: float
) -> float:
    """Utility of a terminal state, nudged so that quicker wins and slower losses are preferred.

    :param game: Game problem providing :meth:`GameProblem.utility`.
    :param state: Terminal state.
    :param root_player: Player the search is run for.
    :param ply: Number of moves played from the root to reach ``state``.
    :param epsilon: Utility given up per ply.
    :return: Adjusted utility (draws are left untouched).
    """
    utility = game.utility(state, root_player)
    if utility == 1:
        return utility - ply * epsilon
    if utility == 0:
        return utility + ply * epsilon
    return utility


def advanced_heuristic_alphabeta_search(
    game: GameProblem,
    state: GameState,
    eval_fn: Callable[[GameState, Any], float],
    cutoff_test: Callable[[GameState, int, float], bool],
    killer_moves: Optional[Dict[int, List[Any]]] = None,
    transposition_table: Optional[Dict[Hashable, TTEntry]] = None,
    order_moves_callback: Optional[Callable[[Any, List[Any]], List[Any]]] = None,
    max_depth: int = 3,
    start_time: Optional[float] = None,
    max_tt_entries: int = DEFAULT_MAX_TT_ENTRIES,
    epsilon: float = DEPTH_PREFERENCE_EPSILON,
    noisy_moves: Optional[Callable[[GameState], List[Any]]] = None,
    quiescence_plies: int = 0,
) -> Tuple[Any, SearchStatistics]:
    """Alpha-beta search with transposition table, killer moves and move ordering.

    Differences with :func:`heuristic_alphabeta_search`:

    * depth is counted **downwards** (``depth_remaining``): the root has ``max_depth`` and the
      search is hard-limited to ``max_depth`` plies;
    * results are cached in ``transposition_table`` and reused by later searches (also by the
      next move of a game);
    * moves that caused a cut-off are tried first (killer moves, captures included), followed by the
      optional ``order_moves_callback``;
    * terminal values are adjusted by ``epsilon`` per ply to prefer fast wins and slow losses.

    If ``cutoff_test`` returns True while ``depth_remaining > 0`` (typically because time ran
    out) the search is flagged as truncated in the returned statistics and **no** transposition
    entry is stored from then on, because the values computed under a cut are not trustworthy.

    :param game: The game problem instance containing rules and utility function
    :type game: GameProblem
    :param state: The current game state to analyze
    :type state: GameState
    :param eval_fn: Heuristic evaluation function ``(state, player) -> float`` in [0, 1]
    :type eval_fn: Callable[[GameState, Any], float]
    :param cutoff_test: ``(state, depth_remaining, elapsed_seconds) -> bool``
    :type cutoff_test: Callable[[GameState, int, float], bool]
    :param killer_moves: Mapping ``depth_remaining -> killer moves``, shared between searches
    :type killer_moves: Optional[Dict[int, List[Any]]]
    :param transposition_table: Table shared between searches
    :type transposition_table: Optional[Dict[Hashable, TTEntry]]
    :param order_moves_callback: ``(board, moves) -> moves`` applied after the killer ordering
    :type order_moves_callback: Optional[Callable[[Any, List[Any]], List[Any]]]
    :param max_depth: Number of plies to search
    :type max_depth: int
    :param start_time: ``time.time()`` value ``elapsed`` is measured from (default: call time).
        Iterative deepening passes the start of the move so the clock does not restart
        at every iteration.
    :type start_time: Optional[float]
    :param max_tt_entries: Size cap of the transposition table
    :type max_tt_entries: int
    :param epsilon: Utility given up per ply when preferring faster wins
    :type epsilon: float
    :param noisy_moves: ``state -> moves`` searched by the quiescence search (for chess, captures
        and promotions). Quiescence is off when it is None.
    :type noisy_moves: Optional[Callable[[GameState], List[Any]]]
    :param quiescence_plies: Maximum plies of quiescence search below the horizon (0 = off)
    :type quiescence_plies: int
    :return: ``(best_move, statistics)``; ``best_move`` is None if the root itself was cut off
    :rtype: Tuple[Any, SearchStatistics]
    """
    root_player = state.get_current_player()
    time_of_start = time.time() if start_time is None else start_time
    killers = {} if killer_moves is None else killer_moves
    table = {} if transposition_table is None else transposition_table

    nodes_visited = 0
    pruning_count = 0
    max_ply_reached = 0
    truncated = False

    def visit(
        node: GameState, alpha: float, beta: float, depth_remaining: int
    ) -> Tuple[Optional[AdversarialSearchResult], float, float]:
        """Common entry of ``max_value`` and ``min_value``.

        Returns a result when the node can be answered immediately (transposition hit, terminal
        state or cut-off), together with the possibly tightened ``(alpha, beta)`` window.
        """
        nonlocal nodes_visited, max_ply_reached, truncated
        nodes_visited += 1
        ply = max_depth - depth_remaining
        max_ply_reached = max(max_ply_reached, ply)

        hit, alpha, beta = probe_transposition_table(
            table, node.key(), depth_remaining, alpha, beta
        )
        if hit is not None:
            return hit, alpha, beta

        if game.is_terminal(node):
            utility = _terminal_utility(game, node, root_player, ply, epsilon)
            return AdversarialSearchResult(utility, None), alpha, beta

        if depth_remaining <= 0:
            if noisy_moves is not None and quiescence_plies > 0:
                value = quiescence(node, alpha, beta, quiescence_plies, ply)
                return AdversarialSearchResult(value, None), alpha, beta
            return AdversarialSearchResult(eval_fn(node, root_player), None), alpha, beta

        if cutoff_test(node, depth_remaining, time.time() - time_of_start):
            truncated = True
            return AdversarialSearchResult(eval_fn(node, root_player), None), alpha, beta

        return None, alpha, beta

    def quiescence(node: GameState, alpha: float, beta: float, plies_left: int, ply: int) -> float:
        """Value of a horizon node searching only noisy moves (stand-pat, no table)."""
        nonlocal nodes_visited, pruning_count
        best_value = eval_fn(node, root_player)  # stand-pat: nobody is forced to capture
        if plies_left <= 0:
            return best_value
        maximizing = node.get_current_player() == root_player
        if maximizing:
            alpha = max(alpha, best_value)
        else:
            beta = min(beta, best_value)
        if alpha >= beta:
            pruning_count += 1
            return best_value
        for action in noisy_moves(node):
            node.apply_move(action)
            try:
                nodes_visited += 1
                if game.is_terminal(node):
                    value = _terminal_utility(game, node, root_player, ply + 1, epsilon)
                else:
                    value = quiescence(node, alpha, beta, plies_left - 1, ply + 1)
            finally:
                node.undo_move()
            if maximizing:
                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
            else:
                best_value = min(best_value, value)
                beta = min(beta, best_value)
            if alpha >= beta:
                pruning_count += 1
                break
        return best_value

    def ordered_actions(node: GameState, depth_remaining: int) -> List[Any]:
        """Legal moves, killer moves first and then the optional callback ordering."""
        actions = reorder_by_killer_moves(game.actions(node), depth_remaining, killers)
        if order_moves_callback:
            actions = order_moves_callback(node.get_board(), actions)
        return actions

    def finish(
        node: GameState,
        value: float,
        move: Any,
        depth_remaining: int,
        alpha_orig: float,
        beta_orig: float,
    ) -> AdversarialSearchResult:
        """Common exit of ``max_value`` and ``min_value``: store the result and return it."""
        result = AdversarialSearchResult(value, move)
        if not truncated:
            store_transposition_entry(
                table, node.key(), result, depth_remaining, alpha_orig, beta_orig, max_tt_entries
            )
        return result

    def max_value(
        node: GameState, alpha: float, beta: float, depth_remaining: int
    ) -> AdversarialSearchResult:
        """Value of ``node`` when the root player moves (maximizing)."""
        nonlocal pruning_count
        alpha_orig, beta_orig = alpha, beta
        early, alpha, beta = visit(node, alpha, beta, depth_remaining)
        if early is not None:
            return early

        best_value = float("-inf")
        best_move = None
        for action in ordered_actions(node, depth_remaining):
            node.apply_move(action)
            try:
                child = min_value(node, alpha, beta, depth_remaining - 1)
            finally:
                node.undo_move()
            if child.value > best_value:
                best_value = child.value
                best_move = action
                alpha = max(alpha, best_value)
            if alpha >= beta:
                pruning_count += 1
                record_killer_move(killers, depth_remaining, action)
                break

        return finish(node, best_value, best_move, depth_remaining, alpha_orig, beta_orig)

    def min_value(
        node: GameState, alpha: float, beta: float, depth_remaining: int
    ) -> AdversarialSearchResult:
        """Value of ``node`` when the opponent moves (minimizing)."""
        nonlocal pruning_count
        alpha_orig, beta_orig = alpha, beta
        early, alpha, beta = visit(node, alpha, beta, depth_remaining)
        if early is not None:
            return early

        best_value = float("inf")
        best_move = None
        for action in ordered_actions(node, depth_remaining):
            node.apply_move(action)
            try:
                child = max_value(node, alpha, beta, depth_remaining - 1)
            finally:
                node.undo_move()
            if child.value < best_value:
                best_value = child.value
                best_move = action
                beta = min(beta, best_value)
            if alpha >= beta:
                pruning_count += 1
                record_killer_move(killers, depth_remaining, action)
                break

        return finish(node, best_value, best_move, depth_remaining, alpha_orig, beta_orig)

    root_result = max_value(state, float("-inf"), float("inf"), max_depth)
    statistics = SearchStatistics(nodes_visited, pruning_count, max_ply_reached, truncated)
    return root_result.move, statistics


def heuristic_alphabeta_iterative_deepening_search(
    game: GameProblem,
    state: GameState,
    eval_fn: Callable[[GameState, Any], float],
    cutoff_test: Callable[[GameState, int, float], bool],
    killer_moves: Optional[Dict[int, List[Any]]] = None,
    transposition_table: Optional[Dict[Hashable, TTEntry]] = None,
    order_moves_callback: Optional[Callable[[Any, List[Any]], List[Any]]] = None,
    max_depth: int = 3,
    time_limit: float = 2.0,
    noisy_moves: Optional[Callable[[GameState], List[Any]]] = None,
    quiescence_plies: int = 0,
    max_tt_entries: int = DEFAULT_MAX_TT_ENTRIES,
) -> Tuple[Any, SearchStatistics]:
    """Iterative deepening over :func:`advanced_heuristic_alphabeta_search`.

    Searches depth 1, 2, ... ``max_depth`` and returns the move of the deepest **completed**
    iteration. Before starting an iteration the time it would take is estimated from the
    two previous ones (growth factor = ratio of their durations) and the loop stops if it does
    not fit in the remaining time. An iteration interrupted by ``cutoff_test`` is discarded
    unless it is the very first one (some move is always needed).

    :param game: The game problem instance containing rules and utility function
    :type game: GameProblem
    :param state: The current game state to analyze
    :type state: GameState
    :param eval_fn: Heuristic evaluation function ``(state, player) -> float``
    :type eval_fn: Callable[[GameState, Any], float]
    :param cutoff_test: ``(state, depth_remaining, elapsed_since_move_start) -> bool``
    :type cutoff_test: Callable[[GameState, int, float], bool]
    :param killer_moves: Killer moves shared across iterations
    :type killer_moves: Optional[Dict[int, List[Any]]]
    :param transposition_table: Table shared across iterations and moves
    :type transposition_table: Optional[Dict[Hashable, TTEntry]]
    :param order_moves_callback: Optional move ordering callback
    :type order_moves_callback: Optional[Callable[[Any, List[Any]], List[Any]]]
    :param max_depth: Deepest iteration to run
    :type max_depth: int
    :param time_limit: Soft time budget in seconds
    :type time_limit: float
    :param noisy_moves: Moves searched by the quiescence search (see the advanced search)
    :type noisy_moves: Optional[Callable[[GameState], List[Any]]]
    :param quiescence_plies: Maximum plies of quiescence search (0 = off)
    :type quiescence_plies: int
    :param max_tt_entries: Cap on the transposition table size (see the advanced search)
    :type max_tt_entries: int
    :return: ``(best_move, statistics)`` with node and pruning counts summed over all iterations
    :rtype: Tuple[Any, SearchStatistics]
    """
    start_time = time.time()
    overshoot_factor = 1.15  # tolerate slightly exceeding the budget to complete a depth
    killers = {} if killer_moves is None else killer_moves
    table = {} if transposition_table is None else transposition_table

    best_move = None
    total_nodes = 0
    total_prunings = 0
    deepest_ply = 0
    completed_depth = 0
    last_truncated = False
    iteration_times: List[float] = []

    for depth in range(1, max_depth + 1):
        remaining = time_limit - (time.time() - start_time)

        if len(iteration_times) >= 2:
            last, previous = iteration_times[-1], iteration_times[-2]
            growth = last / previous if previous > 1e-6 and last > 1e-6 else 1.5
            if last * growth > remaining * overshoot_factor:
                break

        iteration_start = time.time()
        move, stats = advanced_heuristic_alphabeta_search(
            game=game,
            state=state,
            eval_fn=eval_fn,
            cutoff_test=cutoff_test,
            killer_moves=killers,
            transposition_table=table,
            order_moves_callback=order_moves_callback,
            max_depth=depth,
            start_time=start_time,
            noisy_moves=noisy_moves,
            quiescence_plies=quiescence_plies,
            max_tt_entries=max_tt_entries,
        )
        iteration_times.append(time.time() - iteration_start)
        total_nodes += stats.nodes_visited
        total_prunings += stats.pruning_count
        last_truncated = stats.truncated
        deepest_ply = max(deepest_ply, stats.max_depth_reached)

        if stats.truncated:
            if best_move is None and move is not None:
                best_move = move  # better a partial answer than none at all
            break

        best_move = move
        completed_depth = depth
        if time.time() - start_time >= time_limit:
            break

    final_stats = SearchStatistics(
        total_nodes,
        total_prunings,
        deepest_ply,
        last_truncated and best_move is None,
        completed_depth,
    )
    return best_move, final_stats
