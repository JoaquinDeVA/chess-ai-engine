"""
Heuristic Alpha-Beta Chess Bot

A chess bot that uses alpha-beta pruning with heuristic evaluation functions
for positions that are not terminal states.
"""
from typing import Dict, Callable

import chess

from adversarial_search.chess_game_state import ChessGameState
from adversarial_search.game_problems import GameState

from adversarial_search.chess_problem import ChessProblem
from adversarial_search.game_algorithms import heuristic_alphabeta_search
from bots.chess_bot import ChessBot
from bots.features import get_position_features



def linear_weighted_heuristic(state: ChessGameState, player: chess.Color, weights: Dict[str, float]) -> float:
    """
    Generic linear weighted heuristic evaluation function.

    Collects position features, applies weighted linear combination, and clips the result.

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :param weights: Dictionary of feature weights (must sum to 1.0)
    :type weights: Dict[str, float]
    :return: Evaluation score (0 to 1)
    :rtype: float
    """
    board = state.get_board()
    features = get_position_features(board)
    score = weighted_linear_features(features, weights, player)
    return max(1e-2, min(1.0 - 1e-2, score))


def weighted_linear_features(features: Dict[str, float], weights: Dict[str, float], player: chess.Color) -> float:
    """
    Generic weighted linear combination of normalized features.

    :param features: Dictionary of features from get_position_features
    :type features: Dict[str, float]
    :param weights: Dictionary of feature weights (must sum to 1.0)
    :type weights: Dict[str, float]
    :param player: Player to evaluate for (chess.WHITE or chess.BLACK)
    :type player: chess.Color
    :return: Weighted evaluation score (0 to 1)
    :rtype: float
    :raises ValueError: If weights don't sum to 1.0 or contain invalid keys
    """
    # Validate weights
    normalized_features = {
        "material_diff_normalized",
        "positional_diff_normalized",
        "mobility_diff_normalized",
        "center_diff_normalized",
        "king_safety_diff_normalized",
        "development_diff_normalized"
    }

    # Check that all weight keys are valid normalized features
    invalid_keys = set(weights.keys()) - normalized_features
    if invalid_keys:
        raise ValueError(f"Invalid weight keys: {invalid_keys}. Valid keys: {normalized_features}")

    # Check that weights sum to approximately 1.0
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    # Check that all weights are non-negative
    negative_weights = {k: v for k, v in weights.items() if v < 0}
    if negative_weights:
        raise ValueError(f"Weights must be non-negative, got negative weights: {negative_weights}")

    # Calculate weighted score
    if player == chess.WHITE:
        score = sum(weights.get(feature, 0.0) * features[feature] for feature in normalized_features)
    else:
        # For black, invert the features (1.0 - feature_value)
        score = sum(weights.get(feature, 0.0) * (1.0 - features[feature]) for feature in normalized_features)

    return score



def h1(state: ChessGameState, player: chess.Color) -> float:
    """
    Heuristic 1: Equal-weighted evaluation.

    Uses uniform weights (1/6 ≈ 16.67% each) for all six normalized features.

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :return: Evaluation score (0 to 1)
    :rtype: float
    """
    equal_weight = 1.0 / 6
    weights = {
        "material_diff_normalized": equal_weight,
        "positional_diff_normalized": equal_weight,
        "mobility_diff_normalized": equal_weight,
        "center_diff_normalized": equal_weight,
        "king_safety_diff_normalized": equal_weight,
        "development_diff_normalized": equal_weight
    }
    return linear_weighted_heuristic(state, player, weights)


def h2(state: ChessGameState, player: chess.Color) -> float:
    """
    Heuristic 2

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :return: Evaluation score (0 to 1)
    :rtype: float
    """
    weights = {
        "material_diff_normalized": 0.1,
        "positional_diff_normalized": 0.1,
        "mobility_diff_normalized": 0.0,
        "center_diff_normalized": 0.0,
        "king_safety_diff_normalized": 0.5,
        "development_diff_normalized": 0.3
    }
    return linear_weighted_heuristic(state, player, weights)


def h3(state: ChessGameState, player: chess.Color) -> float:
    """
    Heuristic 3:

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :return: Evaluation score (0 to 1)
    :rtype: float
    """
    weights = {
        "material_diff_normalized": 0.5,
        "positional_diff_normalized": 0.2,
        "mobility_diff_normalized": 0.2,
        "center_diff_normalized": 0.0,
        "king_safety_diff_normalized": 0.1,
        "development_diff_normalized": 0.0
    }
    return linear_weighted_heuristic(state, player, weights)

def h4(state: ChessGameState, player: chess.Color) -> float:

    #Necesita mucho refinamiento

    board = state.get_board()
    features = get_position_features(board)


    #Calculamos en que fase del juego estamos
    total_material = features["white_material"] + features["black_material"]
    progress = max(0.0, (total_material - 12) / (72 - 12))

    opening_weights = {
        "material_diff_normalized": 0.25,
        "positional_diff_normalized": 0.15,
        "mobility_diff_normalized": 0.25,
        "center_diff_normalized": 0.20,
        "king_safety_diff_normalized": 0.10,
        "development_diff_normalized": 0.05
    }

    midgame_weights = {
        "material_diff_normalized": 0.35,
        "positional_diff_normalized": 0.15,
        "mobility_diff_normalized": 0.25,
        "center_diff_normalized": 0.10,
        "king_safety_diff_normalized": 0.10,
        "development_diff_normalized": 0.05
    }

    endgame_weights = {
        "material_diff_normalized": 0.50,
        "positional_diff_normalized": 0.15,
        "mobility_diff_normalized": 0.20,
        "center_diff_normalized": 0.05,
        "king_safety_diff_normalized": 0.05,
        "development_diff_normalized": 0.05
    }

    weights = interpolate_weights(opening_weights,midgame_weights, endgame_weights, progress)

    #Ajustamos en base a diferentes situaciones

    hanging_count = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == player:
            attackers = board.attackers(not player, square)
            defenders = board.attackers(player, square)
            if len(attackers) > len(defenders):
                hanging_count += 1

    if hanging_count > 0:
        # Increase material importance
        weights["material_diff_normalized"] += 0.1 * hanging_count


    #Normalizamos los valores
    total = sum(weights.values())
    normalized_weights = {key: weights[key] / total for key in weights}

    return linear_weighted_heuristic(state, player, normalized_weights)


def interpolate_weights(opening_weights: Dict[str, float],
                        midgame_weights: Dict[str, float],
                        endgame_weights: Dict[str, float],
                        progress_of_game : float) -> Dict[str, float]:
    """
     Linearly interpolate between two weight sets.

     :param opening_weights: Weights for opening/midgame
     :type opening_weights: Dict[str, float]
     :param midgame_weights: Weights for midgame
     :type midgame_weights: Dict[str, float]
     :param endgame_weights: Weights for endgame
     :type endgame_weights: Dict[str, float]
     :param progress_of_game: Phase factor in [0,1], 1=open, 0=end
     :type progress_of_game: float
     :return: Interpolated weights dict
     """
    #Transicion entre fases
    OPPENING_THRESHOLD = 0.7
    MIDGAME_THRESHOLD = 0.2

    if progress_of_game > OPPENING_THRESHOLD:
        return opening_weights

    if progress_of_game < MIDGAME_THRESHOLD:
        return endgame_weights

    if progress_of_game > 0.5:
        #Pasamos de la escala 0.5 - 0.7 a 0.0 - 1.0
        transition_progress = (progress_of_game - 0.5) / (OPPENING_THRESHOLD - 0.5)
        return {
            key: transition_progress * opening_weights[key] +
                 (1 - transition_progress) * midgame_weights[key]
            for key in opening_weights
        }

    #Solo queda que estemos entre 0.0 - 0.3
    transition_progress = (progress_of_game - MIDGAME_THRESHOLD) / (0.5 - MIDGAME_THRESHOLD)

    return {
            key: transition_progress * midgame_weights[key] +
                 (1 - transition_progress) * endgame_weights[key]
            for key in opening_weights
            }


class HeuristicAlphaBetaBot(ChessBot):
    """
    Chess bot using heuristic alpha-beta search with configurable parameters.
    """

    def __init__(self, eval_function: Callable[[ChessGameState, chess.Color], float]=h3,
                 max_depth: int = 3, problem : ChessProblem = None, name : str = None) -> None:
        """
        Initialize the heuristic alpha-beta bot.

        :param eval_function: Evaluation function (state, player) -> float
        :type eval_function: Callable[[ChessGameState, chess.Color], float]
        :param max_depth: Maximum search depth
        :type max_depth: int
        """
        if name is None:
            name = "HeuristicAlphaBetaBot"
        super().__init__(name, "Joaquin De Vicente Abad")
        self.eval_function = eval_function
        self.max_depth = max_depth
        self.problem = problem if problem is not None else ChessProblem(initial_board= None)


    def get_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """
        Get the best move using heuristic alpha-beta search.

        :param board: Current chess position
        :type board: chess.Board
        :param time_limit: Maximum time to think in seconds
        :type time_limit: float
        :return: Best move found
        :rtype: chess.Move
        """

        state = ChessGameState(board= board)


        return heuristic_alphabeta_search( game= self.problem ,
                                          state= state,
                                          eval_fn= self.eval_function,
                                          cutoff_test= self.cutoff_test
                                          )



    def cutoff_test(self, state : GameState , depth : int, time_used : float) -> bool:

        return depth >= self.max_depth




