"""Evaluations for the Rust core: the native version of a model, or a callback to Python."""

import logging
from typing import Callable

import chess
import numpy as np

from chess_engine import _core
from chess_engine.evaluation.heuristics import FEATURE_NAMES, h1, load_h1_weights
from chess_engine.search.chess_game_state import ChessGameState

from .base import EvaluationModel
from .evaluator import ModelEvaluator
from .linear_eval import LinearEvalModel
from .neural_eval import NeuralEvalModel

logger = logging.getLogger(__name__)


def h1_evaluator() -> "_core.Evaluator":
    """The heuristic ``h1`` as a native evaluation.

    :return: Evaluator with the fitted weights of ``h1_weights.json``.
    """
    weights = load_h1_weights()
    return _core.Evaluator.h1([weights[name] for name in FEATURE_NAMES])


def callback_evaluator(
    function: Callable[[ChessGameState, chess.Color], float],
) -> "_core.Evaluator":
    """An evaluation computed in Python (slow: the core calls back for every leaf).

    :param function: ``(state, player) -> float``, like ``h1`` or a ``ModelEvaluator``.
    :return: Evaluator that calls ``function``.
    """

    def evaluate(fen: str, white: bool) -> float:
        return function(ChessGameState(chess.Board(fen)), chess.WHITE if white else chess.BLACK)

    return _core.Evaluator.callback(evaluate)


def model_evaluator(model: EvaluationModel) -> "_core.Evaluator":
    """The native evaluation of a model.

    Linear and sparse-network models whose feature set is implemented in the core run inside it;
    any other model (the Keras networks of the course, or a new feature set that has no Rust
    version yet) is evaluated through a callback to Python, which works but is slow.

    :param model: A loaded model.
    :return: Evaluator that gives the same values as ``ModelEvaluator(model)``.
    """
    sparse = isinstance(model, LinearEvalModel)
    if sparse and model.features.name not in _core.FEATURE_SETS:
        logger.warning(
            "the feature set %r has no Rust version: %s is evaluated through Python (slow)",
            model.features.name,
            model.name,
        )
        return callback_evaluator(ModelEvaluator(model))
    if isinstance(model, NeuralEvalModel):
        layers = [
            (
                np.ascontiguousarray(weights, dtype=np.float64),
                np.ascontiguousarray(bias, dtype=np.float64),
            )
            for weights, bias in model.layers
        ]
        return _core.Evaluator.network(model.features.name, model.rep.perspective, layers)
    if isinstance(model, LinearEvalModel):
        weights = np.ascontiguousarray(model.weights, dtype=np.float64)
        return _core.Evaluator.linear(
            model.features.name, model.rep.perspective, weights, model.bias
        )
    return callback_evaluator(ModelEvaluator(model))


def heuristic_evaluator(
    function: Callable[[ChessGameState, chess.Color], float],
) -> "_core.Evaluator":
    """The native evaluation of a heuristic function (``h1`` runs natively, others by callback).

    :param function: ``(state, player) -> float``.
    :return: Evaluator.
    """
    return h1_evaluator() if function is h1 else callback_evaluator(function)
