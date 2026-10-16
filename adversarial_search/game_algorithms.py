import random
import time
from typing import Any, NamedTuple, Callable, List, Dict, Hashable

import chess
from chess.syzygy import Tablebase

from .chess_game_state import ChessGameState
from .game_problems import GameProblem, GameState


def random_moves(game: "GameProblem", state: Any) -> Any:
    """
    A player that chooses a legal move at random.

    This strategy function randomly selects one of the available legal moves
    from the current game state. It provides no strategic consideration and
    is primarily useful for testing, baseline comparisons, or educational purposes.

    :param game: The game problem instance containing rules and available actions
    :type game: GameProblem
    :param state: The current game state to choose a move from
    :type state: Any
    :return: A randomly selected legal move
    :rtype: Any
    """
    return random.choice(game.actions(state))


class AdversarialSearchResult(NamedTuple):
    """
    Result of adversarial search (minimax or alpha-beta) containing utility value and best move.

    :param value: The utility value of the position for the searching player
    :type value: float
    :param move: The best move to make, or None if terminal state
    :type move: Any | None
    """

    value: float
    move: Any | None


def minimax_search(game: "GameProblem", state: "GameState") -> Any:
    """
    Search game tree to determine best move using minimax algorithm.

    The minimax algorithm searches the complete game tree to find the optimal
    move by assuming both players play perfectly. It alternates between
    maximizing and minimizing players at each level of the tree.

    :param game: The game problem instance containing rules and utility function
    :type game: GameProblem
    :param state: The current game state to analyze
    :type state: GameState
    :return: The best move for the current player
    :rtype: Any
    """
    player = state.get_current_player()

    def max_value(state: "GameState") -> AdversarialSearchResult:
        """
        Calculate the maximum value achievable from this state.

        :param state: The game state to evaluate
        :type state: GameState
        :return: The maximum value and corresponding best move
        :rtype: AdversarialSearchResult
        """
        if game.is_terminal(state):
            return AdversarialSearchResult(game.utility(state, player), None)

        best_value = float("-inf")
        best_move = None

        for action in game.actions(state):
            result = min_value(game.result(state, action))
            if result.value > best_value:
                best_value = result.value
                best_move = action

        return AdversarialSearchResult(best_value, best_move)

    def min_value(state: "GameState") -> AdversarialSearchResult:
        """
        Calculate the minimum value achievable from this state.

        :param state: The game state to evaluate
        :type state: GameState
        :return: The minimum value and corresponding best move
        :rtype: AdversarialSearchResult
        """
        if game.is_terminal(state):
            return AdversarialSearchResult(game.utility(state, player), None)

        best_value = float("inf")
        best_move = None

        for action in game.actions(state):
            result = max_value(game.result(state, action))
            if result.value < best_value:
                best_value = result.value
                best_move = action

        return AdversarialSearchResult(best_value, best_move)

    search_result = max_value(state)
    return search_result.move


def alphabeta_search(game: "GameProblem", state: "GameState") -> Any:
    """
    Search game tree to determine best move using alpha-beta pruning.

    Alpha-beta pruning is an optimization of the minimax algorithm that
    eliminates branches that cannot possibly influence the final decision.
    It maintains two values: alpha (best value for maximizing player) and
    beta (best value for minimizing player) to prune unnecessary branches.

    :param game: The game problem instance containing rules and utility function
    :type game: GameProblem
    :param state: The current game state to analyze
    :type state: GameState
    :return: The best move for the current player
    :rtype: Any

    ... see also::
       Based on Figure 5.7 from "Artificial Intelligence: A Modern Approach"
    """
    player = state.get_current_player()

    def max_value(
        state: "GameState", alpha: float, beta: float
    ) -> AdversarialSearchResult:
        """
        Calculate the maximum value achievable from this state with alpha-beta pruning.

        :param state: The game state to evaluate
        :type state: GameState
        :param alpha: The best value the maximizing player can guarantee so far
        :type alpha: float
        :param beta: The best value the minimizing player can guarantee so far
        :type beta: float
        :return: The maximum value and corresponding best move
        :rtype: AdversarialSearchResult
        """
        if game.is_terminal(state):
            return AdversarialSearchResult(game.utility(state, player), None)

        best_value = float("-inf")
        best_move = None

        for action in game.actions(state):
            result = min_value(game.result(state, action), alpha, beta)
            if result.value > best_value:
                best_value = result.value
                best_move = action
                alpha = max(alpha, best_value)

            if alpha >= beta:
                return AdversarialSearchResult(best_value, best_move)

        return AdversarialSearchResult(best_value, best_move)

    def min_value(
        state: "GameState", alpha: float, beta: float
    ) -> AdversarialSearchResult:
        """
        Calculate the minimum value achievable from this state with alpha-beta pruning.

        :param state: The game state to evaluate
        :type state: GameState
        :param alpha: The best value the maximizing player can guarantee so far
        :type alpha: float
        :param beta: The best value the minimizing player can guarantee so far
        :type beta: float
        :return: The minimum value and corresponding best move
        :rtype: AdversarialSearchResult
        """
        if game.is_terminal(state):
            return AdversarialSearchResult(game.utility(state, player), None)

        best_value = float("inf")
        best_move = None

        for action in game.actions(state):
            result = max_value(game.result(state, action), alpha, beta)
            if result.value < best_value:
                best_value = result.value
                best_move = action
                beta = min(beta, best_value)

            if alpha >= beta:
                return AdversarialSearchResult(best_value, best_move)

        return AdversarialSearchResult(best_value, best_move)

    search_result = max_value(state, float("-inf"), float("inf"))
    return search_result.move


class SearchStatistics(NamedTuple):
    """
    Statistics collected during alpha-beta search.

    :param nodes_visited: Total number of nodes evaluated
    :type nodes_visited: int
    :param pruning_count: Number of times alpha-beta pruning occurred
    :type pruning_count: int
    :param max_depth_reached: Maximum depth reached in search
    :type max_depth_reached: int
    """

    nodes_visited: int
    pruning_count: int
    max_depth_reached: int

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
            result = min_value(game.result(state, action), alpha, beta, depth+1)
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
            actions = order_moves_callback(state.get_board() , actions)

        for action in actions:
            result = max_value(game.result(state, action), alpha, beta,depth +1)
            if result.value < best_value:
                best_value = result.value
                best_move = action
                beta = min(beta, best_value)

            if alpha >= beta:
                pruning_count += 1
                return AdversarialSearchResult(best_value, best_move)

        return AdversarialSearchResult(best_value, best_move)


    time_of_start = time.time()
    search_result = max_value(state, float("-inf"), float("inf"), depth = 0)
    return search_result.move, SearchStatistics(nodes_visited, pruning_count, max_depth_reached)




class TTEntry:
    """Entry for the transposition table"""
    def __init__(self, result : AdversarialSearchResult, depth_remaining: int, flag: str):

        self.result = result # El movimiento
        self.depth_remaining = depth_remaining  # Profundidad restante a la hora de evaluar
        self.flag = flag  # 'EXACT', 'LOWERBOUND', 'UPPERBOUND'. Para poder almacenar información de poda




def advanced_heuristic_alphabeta_search(
    game: GameProblem,
    state: GameState,
    eval_fn: Callable[[GameState, Any], float],
    cutoff_test: Callable[[GameState, int, float], bool],
    killer_moves: Dict[int, List[chess.Move]] = None,
    transposition_table: Dict[Hashable, TTEntry] = None,
    order_moves_callback: Callable[[chess.Board, list[chess.Move]], list[chess.Move]] | None = None,
    max_depth: int = 3
) -> Any:
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
    :param killer_moves: Killer moves to use for evaluation
    :type killer_moves: Dict[int, List[chess.Move]]
    :param transposition_table: Transposition table to store already calculated moves
    :type transposition_table: Dict[Hashable, TTEntry]
    :param order_moves_callback: Callback function for ordering moves
    :type Callable[[chess.Board, list[chess.Move]], list[chess.Move]] | None
    :param max_depth: The maximum depth to search, used for transposition storing
    :type max_depth: int
    :return: The best move for the current player
    :rtype: Any
    """

    root_player = state.get_current_player()
    time_of_start = time.time()

    # Parámetro para ajustar la prioridad de movimientos profundos
    epsilon = 0.003

    # Diccionario para almacenar movimientos tipo killer_moves
    if killer_moves is None:
        killer_moves = {}

    # Diccionario para almacenar los movimientos previamente calculados
    if transposition_table is None:
        transposition_table = {}


    def max_value(
        state: "GameState", alpha: float, beta: float, depth_remaining: int
    ) -> AdversarialSearchResult:
        """
        Calculate the maximum value achievable from this state with alpha-beta pruning.

        :param state: The game state to evaluate
        :type state: GameState
        :param alpha: The best value the maximizing player can guarantee so far
        :type alpha: float
        :param beta: The best value the minimizing player can guarantee so far
        :type beta: float
        :param depth_remaining: Current search depth (root = 0).
        :type depth_remaining: int
        :return: The maximum value and corresponding best move
        :rtype: AdversarialSearchResult
        """

        key = state.key()
        alpha_orig = alpha
        beta_orig = beta

        # Comprobamos si la posición ya ha sido calculada
        entry = transposition_table.get(key)

        #Miramos si ya está guardado, pero con un valor exacto o información de estado (alpha, beta)
        if entry and entry.depth_remaining >= depth_remaining:
            if entry.flag == 'EXACT':
                return entry.result
            elif entry.flag == 'LOWERBOUND':
                alpha = max(alpha, entry.result.value)
            elif entry.flag == 'UPPERBOUND':
                beta = min(beta, entry.result.value)
            if alpha >= beta:
                return entry.result

        if game.is_terminal(state):
            utility = game.utility(state, root_player)

            #Ajustamos en base a la profundidad
            if utility == 1:
                utility -= (max_depth - depth_remaining) * epsilon
            elif utility == 0:
                utility += (max_depth - depth_remaining) * epsilon

            return AdversarialSearchResult(utility, None)


        #Comprobamos si debemos seguir explorando
        if cutoff_test(state, depth_remaining, time.time() - time_of_start):

            return AdversarialSearchResult(eval_fn(state, root_player), None)


        best_value = float("-inf")
        best_move = None

        killer_moves.setdefault(depth_remaining, [])
        actions = reorder_by_killer_moves(game.actions(state), depth_remaining, killer_moves)

        if order_moves_callback:
            actions = order_moves_callback(state.get_board() , actions)


        for action in actions:
            result = min_value(game.result(state, action), alpha, beta, depth_remaining - 1)

            if result.value > best_value:
                best_value = result.value
                best_move = action
                alpha = max(alpha, best_value)

            if alpha >= beta:

                if action not in killer_moves[depth_remaining]:
                    killer_moves[depth_remaining].append(action)
                if len(killer_moves[depth_remaining]) > 3:
                    killer_moves[depth_remaining].pop(0)
                break

        if best_value <= alpha_orig:
            flag = 'UPPERBOUND'
        elif best_value >= beta_orig:
            flag = 'LOWERBOUND'
        else:
            flag = 'EXACT'

        transposition_table[key] = TTEntry(
            result=AdversarialSearchResult(best_value, best_move),
            depth_remaining=depth_remaining,
            flag=flag
        )

        return AdversarialSearchResult(best_value, best_move)

    def min_value(
        state: "GameState", alpha: float, beta: float, depth_remaining: int
    ) -> AdversarialSearchResult:
        """
        Calculate the minimum value achievable from this state with alpha-beta pruning.

        :param state: The game state to evaluate
        :type state: GameState
        :param alpha: The best value the maximizing player can guarantee so far
        :type alpha: float
        :param beta: The best value the minimizing player can guarantee so far
        :type beta: float
        :param depth_remaining: Current search depth (root = 0).
        :type depth_remaining: int
        :return: The minimum value and corresponding best move
        :rtype: AdversarialSearchResult
        """

        key = state.key()
        alpha_orig = alpha
        beta_orig = beta

        #Comprobamos si la posición ya ha sido calculada
        entry = transposition_table.get(key)

        # Miramos si ya está guardado, pero con un valor exacto o información de estado (alpha, beta)
        if entry and entry.depth_remaining >= depth_remaining:
            if entry.flag == 'EXACT':
                return entry.result
            elif entry.flag == 'LOWERBOUND':
                alpha = max(alpha, entry.result.value)
            elif entry.flag == 'UPPERBOUND':
                beta = min(beta, entry.result.value)
            if alpha >= beta:
                return entry.result

        if game.is_terminal(state):
            utility = game.utility(state, root_player)

            # Ajustamos en base a la profundidad
            if utility == 1:
                utility -= depth_remaining * epsilon
            elif utility == 0:
                utility += depth_remaining * epsilon

            return AdversarialSearchResult(utility, None)

        if cutoff_test(state, depth_remaining, time.time() - time_of_start):
            return AdversarialSearchResult(eval_fn(state, root_player), None)

        best_value = float("inf")
        best_move = None

        killer_moves.setdefault(depth_remaining, [])
        actions = reorder_by_killer_moves(game.actions(state), depth_remaining, killer_moves)

        if order_moves_callback:
            actions = order_moves_callback(state.get_board() , actions)

        for action in actions:
            result = max_value(game.result(state, action), alpha, beta, depth_remaining - 1)

            if result.value < best_value:
                best_value = result.value
                best_move = action
                beta = min(beta, best_value)

            if alpha >= beta:

                if action not in killer_moves[depth_remaining]:
                    killer_moves[depth_remaining].append(action)
                if len(killer_moves[depth_remaining]) > 3:
                    killer_moves[depth_remaining].pop(0)
                break

        if best_value <= alpha_orig:
            flag = 'UPPERBOUND'
        elif best_value >= beta_orig:
            flag = 'LOWERBOUND'
        else:
            flag = 'EXACT'

        transposition_table[key] = TTEntry(
            result=AdversarialSearchResult(best_value, best_move),
            depth_remaining=depth_remaining,
            flag=flag
        )

        return AdversarialSearchResult(best_value, best_move)


    search_result = max_value(state, float("-inf"), float("inf"), depth_remaining= max_depth)
    return search_result.move




def reorder_by_killer_moves(moves: list[Any], depth: int, diccionario : Dict[int, List[Any]]):

    killer_moves = diccionario.get(depth, [])
    moves.sort(key= lambda move: move not in killer_moves)

    return moves



















def heuristic_alphabeta_iterative_deepening_search(
    game: GameProblem,
    state: GameState,
    eval_fn: Callable[[GameState, Any], float],
    cutoff_test: Callable[[GameState, int, float], bool],
    killer_moves: Dict[int, List[chess.Move]] = None,
    transposition_table: Dict[Hashable, TTEntry] = None,
    order_moves_callback: Callable[[chess.Board, list[chess.Move]], list[chess.Move]] | None = None,
    max_depth: int = 3,
    time_limit: float = 2,

) -> Any:
    """
    Search game tree using iterative deepening over an alpha-beta pruning with heuristic evaluation and cutoff test.

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
    :param killer_moves: Killer moves to use for evaluation
    :type killer_moves: Dict[int, List[chess.Move]]
    :param transposition_table: Transposition table to store already calculated moves
    :type transposition_table: Dict[Hashable, TTEntry]
    :param order_moves_callback: Order moves to use for evaluation
    :type Callable[[chess.Board, list[chess.Move]], list[chess.Move]] | None
    :param max_depth: The maximum depth to search, used for transposition storing
    :type max_depth: int
    :param time_limit: The maximum time to allow search
    :type time_limit: float
    :return: The best move for the current player
    :rtype: Any
    """

    start_time = time.time()

    # Para hacerlo más agresivo con la toma de profundidades
    overshoot_factor = 1.15

    # Diccionario para almacenar movimientos tipo killer_moves
    if killer_moves is None:
        killer_moves = {}

    # Diccionario para almacenar los movimientos previamente calculados
    if transposition_table is None:
        transposition_table = {}

    #Array de tiempos de exploración
    depth_times = []

    for depth in range(1, max_depth + 1):


        elapsed = time.time() - start_time
        remaining = time_limit - elapsed

        if len(depth_times) > 2:
            last = depth_times[-1]
            prev = depth_times[-2]

            #Si el movimiento ya está calculado o es poco explorable a profundidad baja (jaque)
            if prev > 1e-6 and last > 1e-6:
                branching_factor_est = last / prev
            else:
                branching_factor_est = 1.5

            time_estimation = last * branching_factor_est

            if time_estimation  > remaining * overshoot_factor:
                break

        depth_timer = time.time()

        result = advanced_heuristic_alphabeta_search(
                                          game= game,
                                          state= state,
                                          eval_fn= eval_fn,
                                          cutoff_test= cutoff_test,
                                          killer_moves= killer_moves,
                                          transposition_table= transposition_table,
                                          order_moves_callback= order_moves_callback,
                                          max_depth= depth,
                                          )


        depth_times.append(time.time() - depth_timer)

        if (time.time() - start_time) >= time_limit:
            break

    return result



