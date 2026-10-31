"""Evaluations for the Rust core: the heuristic ``h1`` and the native version of a model."""

import json
from pathlib import Path

import numpy as np

from chess_engine import _core

from .base import EvaluationModel
from .linear_eval import LinearEvalModel
from .neural_eval import NeuralEvalModel

#: Names of the features of ``h1``, in the order the core takes their weights.
FEATURE_NAMES = (
    "material_diff_normalized",
    "positional_diff_normalized",
    "mobility_diff_normalized",
    "center_diff_normalized",
    "king_safety_diff_normalized",
    "development_diff_normalized",
)

#: Versioned JSON file with the fitted weights of ``h1``.
H1_WEIGHTS_PATH = Path(__file__).with_name("h1_weights.json")


def load_h1_weights() -> dict:
    """The fitted weights of ``h1``.

    :return: Weight per feature name in :data:`FEATURE_NAMES`.
    :raises ValueError: If the file is not keyed by exactly those names.
    """
    weights = json.loads(H1_WEIGHTS_PATH.read_text())["weights"]
    if set(weights) != set(FEATURE_NAMES):
        raise ValueError(f"{H1_WEIGHTS_PATH}: weights must be keyed by {list(FEATURE_NAMES)}")
    return {name: float(weights[name]) for name in FEATURE_NAMES}


def h1_evaluator() -> "_core.Evaluator":
    """The heuristic ``h1`` as a native evaluation.

    :return: Evaluator with the fitted weights of ``h1_weights.json``.
    """
    weights = load_h1_weights()
    return _core.Evaluator.h1([weights[name] for name in FEATURE_NAMES])


def model_evaluator(model: EvaluationModel) -> "_core.Evaluator":
    """The native evaluation of a model.

    :param model: A linear or sparse-network model.
    :return: Evaluator that runs inside the core.
    :raises ValueError: If the model's feature set has no Rust version.
    """
    if not isinstance(model, LinearEvalModel) or model.features.name not in _core.FEATURE_SETS:
        raise ValueError(f"{model.name} has no Rust evaluation: only the core evaluates models")
    if isinstance(model, NeuralEvalModel):
        layers = [
            (
                np.ascontiguousarray(weights, dtype=np.float64),
                np.ascontiguousarray(bias, dtype=np.float64),
            )
            for weights, bias in model.layers
        ]
        return _core.Evaluator.network(model.features.name, model.rep.perspective, layers)
    weights = np.ascontiguousarray(model.weights, dtype=np.float64)
    return _core.Evaluator.linear(model.features.name, model.rep.perspective, weights, model.bias)
