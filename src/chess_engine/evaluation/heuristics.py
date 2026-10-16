"""Weighted-linear heuristic evaluation functions ``h1`` .. ``h4``.

Every heuristic has the signature ``(state, player) -> float`` and returns the expected
outcome for ``player`` in ``[0.01, 0.99]`` (terminal values are 0, 0.5 and 1, so a heuristic
value never outranks a real win).

``h1``-``h3`` use fixed weights; ``h3t`` uses the weights of ``h3``'s features fitted to
Stockfish (see ``scripts/data/fit_h3t.py``); ``h4`` blends three weight sets (opening, middlegame,
endgame) according to the remaining material and reacts to hanging pieces.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import chess

from chess_engine.search.chess_game_state import ChessGameState

from .features import get_position_features

#: Names of the normalized features a weight dictionary must be keyed by.
FEATURE_NAMES = (
    "material_diff_normalized",
    "positional_diff_normalized",
    "mobility_diff_normalized",
    "center_diff_normalized",
    "king_safety_diff_normalized",
    "development_diff_normalized",
)

#: Lowest/highest value a heuristic may return, to stay clear of the terminal values 0 and 1.
HEURISTIC_MARGIN = 1e-2

H2_WEIGHTS = {
    "material_diff_normalized": 0.1,
    "positional_diff_normalized": 0.1,
    "mobility_diff_normalized": 0.0,
    "center_diff_normalized": 0.0,
    "king_safety_diff_normalized": 0.5,
    "development_diff_normalized": 0.3,
}

H3_WEIGHTS = {
    "material_diff_normalized": 0.5,
    "positional_diff_normalized": 0.2,
    "mobility_diff_normalized": 0.2,
    "center_diff_normalized": 0.0,
    "king_safety_diff_normalized": 0.1,
    "development_diff_normalized": 0.0,
}

H4_OPENING_WEIGHTS = {
    "material_diff_normalized": 0.25,
    "positional_diff_normalized": 0.15,
    "mobility_diff_normalized": 0.25,
    "center_diff_normalized": 0.20,
    "king_safety_diff_normalized": 0.10,
    "development_diff_normalized": 0.05,
}

H4_MIDGAME_WEIGHTS = {
    "material_diff_normalized": 0.35,
    "positional_diff_normalized": 0.15,
    "mobility_diff_normalized": 0.25,
    "center_diff_normalized": 0.10,
    "king_safety_diff_normalized": 0.10,
    "development_diff_normalized": 0.05,
}

H4_ENDGAME_WEIGHTS = {
    "material_diff_normalized": 0.50,
    "positional_diff_normalized": 0.15,
    "mobility_diff_normalized": 0.20,
    "center_diff_normalized": 0.05,
    "king_safety_diff_normalized": 0.05,
    "development_diff_normalized": 0.05,
}

# Game-progress thresholds used by :func:`interpolate_weights` (1.0 = full board, 0.0 = bare kings).
OPENING_THRESHOLD = 0.7
BLEND_MIDPOINT = 0.5
ENDGAME_THRESHOLD = 0.2

# Material range (both sides, without kings) used to compute the game progress in ``h4``.
MIN_MATERIAL = 12
MAX_MATERIAL = 72

#: Extra weight given to material per hanging piece in ``h4``.
HANGING_PIECE_BONUS = 0.1


def weighted_linear_features(
    features: Dict[str, float], weights: Dict[str, float], player: chess.Color
) -> float:
    """Combine normalized features linearly, from ``player``'s point of view.

    For Black the features are inverted (``1 - value``) because they are expressed from
    White's perspective.

    :param features: Dictionary returned by :func:`evaluation.features.get_position_features`
    :type features: Dict[str, float]
    :param weights: Non-negative feature weights that sum to 1
    :type weights: Dict[str, float]
    :param player: Player to evaluate for (``chess.WHITE`` or ``chess.BLACK``)
    :type player: chess.Color
    :return: Weighted score in ``[0, 1]``
    :rtype: float
    :raises ValueError: If a weight key is unknown, a weight is negative or they do not sum to 1
    """
    invalid_keys = set(weights) - set(FEATURE_NAMES)
    if invalid_keys:
        raise ValueError(f"Invalid weight keys: {invalid_keys}. Valid keys: {set(FEATURE_NAMES)}")

    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    negative_weights = {key: value for key, value in weights.items() if value < 0}
    if negative_weights:
        raise ValueError(f"Weights must be non-negative, got negative weights: {negative_weights}")

    if player == chess.WHITE:
        return sum(weights.get(name, 0.0) * features[name] for name in FEATURE_NAMES)
    return sum(weights.get(name, 0.0) * (1.0 - features[name]) for name in FEATURE_NAMES)


def linear_weighted_heuristic(
    state: ChessGameState, player: chess.Color, weights: Dict[str, float]
) -> float:
    """Evaluate ``state`` with a weighted combination of position features.

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :param weights: Feature weights (see :func:`weighted_linear_features`)
    :type weights: Dict[str, float]
    :return: Evaluation clipped to ``[0.01, 0.99]``
    :rtype: float
    """
    features = get_position_features(state.get_board())
    score = weighted_linear_features(features, weights, player)
    return max(HEURISTIC_MARGIN, min(1.0 - HEURISTIC_MARGIN, score))


def h1(state: ChessGameState, player: chess.Color) -> float:
    """Heuristic 1: all six features weighted equally (1/6 each).

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :return: Evaluation in ``[0.01, 0.99]``
    :rtype: float
    """
    weights = {name: 1.0 / len(FEATURE_NAMES) for name in FEATURE_NAMES}
    return linear_weighted_heuristic(state, player, weights)


def h2(state: ChessGameState, player: chess.Color) -> float:
    """Heuristic 2: favours king safety (50%) and piece development (30%).

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :return: Evaluation in ``[0.01, 0.99]``
    :rtype: float
    """
    return linear_weighted_heuristic(state, player, H2_WEIGHTS)


def h3(state: ChessGameState, player: chess.Color) -> float:
    """Heuristic 3: material-centred (50%) with positional and mobility terms (20% each).

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :return: Evaluation in ``[0.01, 0.99]``
    :rtype: float
    """
    return linear_weighted_heuristic(state, player, H3_WEIGHTS)


#: Versioned JSON file with the weights of :func:`h3t`, written by ``scripts/data/fit_h3t.py``.
H3T_WEIGHTS_PATH = Path(__file__).with_name("h3t_weights.json")

_H3T_CACHE: Dict[str, Dict[str, float]] = {}


def load_h3t_weights(path: Optional[Path] = None) -> Dict[str, float]:
    """Load the fitted weights of :func:`h3t`, once per file.

    :param path: Weights file (default: :data:`H3T_WEIGHTS_PATH`)
    :type path: Optional[Path]
    :return: Weight per feature name in :data:`FEATURE_NAMES`
    :rtype: Dict[str, float]
    :raises FileNotFoundError: If the file does not exist (it must be fitted first)
    :raises ValueError: If the weights are not keyed by exactly the feature names
    """
    location = Path(path) if path is not None else H3T_WEIGHTS_PATH
    key = str(location)
    if key not in _H3T_CACHE:
        if not location.exists():
            raise FileNotFoundError(
                f"{location} not found: fit it with scripts/data/fit_h3t.py first"
            )
        weights = json.loads(location.read_text())["weights"]
        if set(weights) != set(FEATURE_NAMES):
            raise ValueError(f"{location}: weights must be keyed by {list(FEATURE_NAMES)}")
        _H3T_CACHE[key] = {name: float(weights[name]) for name in FEATURE_NAMES}
    return _H3T_CACHE[key]


def reset_h3t_cache() -> None:
    """Forget the weights loaded by :func:`load_h3t_weights` (the next call reads the file)."""
    _H3T_CACHE.clear()


def h3t(state: ChessGameState, player: chess.Color) -> float:
    """Tuned classical reference: the features of ``h3`` with weights fitted to Stockfish.

    The White value is ``0.5 + sum(w * (feature - 0.5))``, clipped to ``[0.01, 0.99]``; Black
    gets ``1 - white``, so the two values are exactly complementary. The weights may have any
    sign and need not sum to 1.

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :return: Evaluation in ``[0.01, 0.99]``
    :rtype: float
    :raises FileNotFoundError: If the weights file has not been fitted yet
    """
    weights = load_h3t_weights()
    features = get_position_features(state.get_board())
    white = 0.5 + sum(weights[name] * (features[name] - 0.5) for name in FEATURE_NAMES)
    white = max(HEURISTIC_MARGIN, min(1.0 - HEURISTIC_MARGIN, white))
    return white if player == chess.WHITE else 1.0 - white


def interpolate_weights(
    opening_weights: Dict[str, float],
    midgame_weights: Dict[str, float],
    endgame_weights: Dict[str, float],
    progress_of_game: float,
) -> Dict[str, float]:
    """Blend the opening, middlegame and endgame weight sets by game progress.

    Above ``OPENING_THRESHOLD`` the opening weights are used as they are and below
    ``ENDGAME_THRESHOLD`` the endgame weights; in between the two nearest sets are blended
    linearly, so the result is continuous in ``progress_of_game``.

    :param opening_weights: Weights for the opening
    :type opening_weights: Dict[str, float]
    :param midgame_weights: Weights for the middlegame
    :type midgame_weights: Dict[str, float]
    :param endgame_weights: Weights for the endgame
    :type endgame_weights: Dict[str, float]
    :param progress_of_game: Game progress, 1.0 = full board, 0.0 = bare kings
    :type progress_of_game: float
    :return: Interpolated weight dictionary
    :rtype: Dict[str, float]
    """
    if progress_of_game > OPENING_THRESHOLD:
        return dict(opening_weights)
    if progress_of_game < ENDGAME_THRESHOLD:
        return dict(endgame_weights)

    if progress_of_game > BLEND_MIDPOINT:
        # Opening <-> middlegame: map [0.5, 0.7] onto [0, 1]
        towards_opening = (progress_of_game - BLEND_MIDPOINT) / (OPENING_THRESHOLD - BLEND_MIDPOINT)
        first, second = opening_weights, midgame_weights
    else:
        # Middlegame <-> endgame: map [0.2, 0.5] onto [0, 1]
        towards_opening = (progress_of_game - ENDGAME_THRESHOLD) / (
            BLEND_MIDPOINT - ENDGAME_THRESHOLD
        )
        first, second = midgame_weights, endgame_weights

    return {
        key: towards_opening * first[key] + (1 - towards_opening) * second[key] for key in first
    }


def count_hanging_pieces(board: chess.Board, player: chess.Color) -> int:
    """Count ``player``'s pieces attacked more times than they are defended.

    :param board: Position to inspect
    :type board: chess.Board
    :param player: Owner of the pieces to check
    :type player: chess.Color
    :return: Number of under-defended pieces
    :rtype: int
    """
    hanging = 0
    for square, piece in board.piece_map().items():
        if piece.color != player:
            continue
        if len(board.attackers(not player, square)) > len(board.attackers(player, square)):
            hanging += 1
    return hanging


def h4(state: ChessGameState, player: chess.Color) -> float:
    """Heuristic 4: phase-aware weights, boosted material term when pieces hang.

    The game progress (remaining material) selects between the opening, middlegame and
    endgame weight sets; every hanging piece of ``player`` adds ``HANGING_PIECE_BONUS`` to the
    material weight before the weights are renormalized.

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :return: Evaluation in ``[0.01, 0.99]``
    :rtype: float
    """
    board = state.get_board()
    features = get_position_features(board)

    total_material = features["white_material"] + features["black_material"]
    progress = max(0.0, (total_material - MIN_MATERIAL) / (MAX_MATERIAL - MIN_MATERIAL))

    weights = interpolate_weights(
        H4_OPENING_WEIGHTS, H4_MIDGAME_WEIGHTS, H4_ENDGAME_WEIGHTS, progress
    )

    hanging = count_hanging_pieces(board, player)
    if hanging:
        weights["material_diff_normalized"] += HANGING_PIECE_BONUS * hanging

    total = sum(weights.values())
    normalized = {key: value / total for key, value in weights.items()}
    return linear_weighted_heuristic(state, player, normalized)
