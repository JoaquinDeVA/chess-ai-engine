"""Machine-learning position evaluators.

A model estimates the evaluation of a position in centipawns, from White's point of view, the
way Stockfish would. Models are looked up by name through the registry::

    from chess_engine.models import load_model
    model = load_model("mlp_features")
    model.predict_cp(chess.Board())

See ``docs/ADDING_A_MODEL.md`` to plug in a new one.

Importing this package registers the built-in models; heavy dependencies (Keras)
are only imported when a model is actually loaded or trained.
"""

# Importing the modules below registers the built-in models.
from . import (
    architectures,  # noqa: F401,E402
    linear_eval,  # noqa: F401,E402
    neural_eval,  # noqa: F401,E402
)
from .base import CP_SCALE, EvaluationModel, cp_to_expected_score  # noqa: F401
from .registry import available_models, get_model_class, load_model, register_model  # noqa: F401
