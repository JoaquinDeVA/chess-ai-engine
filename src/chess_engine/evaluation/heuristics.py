"""Weighted-linear heuristic evaluation function ``h1``.

``h1`` has the signature ``(state, player) -> float`` and returns the expected outcome for
``player`` in ``[0.01, 0.99]`` (terminal values are 0, 0.5 and 1, so a heuristic value never
outranks a real win). It is a linear combination of six normalized position features whose
weights are fitted to Stockfish evaluations (see ``scripts/data/fit_h1_weights.py``).
"""

import json
from pathlib import Path
from typing import Dict, Optional

import chess

from chess_engine.evaluation.chess_game_state import ChessGameState

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

#: Versioned JSON file with the weights of :func:`h1`, written by ``scripts/data/fit_h1_weights.py``.
H1_WEIGHTS_PATH = Path(__file__).with_name("h1_weights.json")

_H1_CACHE: Dict[str, Dict[str, float]] = {}


def load_h1_weights(path: Optional[Path] = None) -> Dict[str, float]:
    """Load the fitted weights of :func:`h1`, once per file.

    :param path: Weights file (default: :data:`H1_WEIGHTS_PATH`)
    :type path: Optional[Path]
    :return: Weight per feature name in :data:`FEATURE_NAMES`
    :rtype: Dict[str, float]
    :raises FileNotFoundError: If the file does not exist (it must be fitted first)
    :raises ValueError: If the weights are not keyed by exactly the feature names
    """
    location = Path(path) if path is not None else H1_WEIGHTS_PATH
    key = str(location)
    if key not in _H1_CACHE:
        if not location.exists():
            raise FileNotFoundError(
                f"{location} not found: fit it with scripts/data/fit_h1_weights.py first"
            )
        weights = json.loads(location.read_text())["weights"]
        if set(weights) != set(FEATURE_NAMES):
            raise ValueError(f"{location}: weights must be keyed by {list(FEATURE_NAMES)}")
        _H1_CACHE[key] = {name: float(weights[name]) for name in FEATURE_NAMES}
    return _H1_CACHE[key]


def reset_h1_cache() -> None:
    """Forget the weights loaded by :func:`load_h1_weights` (the next call reads the file)."""
    _H1_CACHE.clear()


def h1(state: ChessGameState, player: chess.Color) -> float:
    """Classical reference: six position features with weights fitted to Stockfish.

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
    weights = load_h1_weights()
    features = get_position_features(state.get_board())
    white = 0.5 + sum(weights[name] * (features[name] - 0.5) for name in FEATURE_NAMES)
    white = max(HEURISTIC_MARGIN, min(1.0 - HEURISTIC_MARGIN, white))
    return white if player == chess.WHITE else 1.0 - white
