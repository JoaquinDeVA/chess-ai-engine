"""Machine-learning position evaluators.

A model estimates the evaluation of a position in centipawns, from White's point of view, the
way Stockfish would. Models are looked up by name through the registry::

    from chess_engine.models import load_model
    model = load_model("linear_pst_terms")
    evaluator = model_evaluator(model)  # from chess_engine.models.native

Importing this package registers the built-in models. It only loads them: the core evaluates
them, and training lives in the laboratory.
"""

# Importing the modules below registers the built-in models.
from . import (
    linear_eval,  # noqa: F401,E402
    neural_eval,  # noqa: F401,E402
)
from .base import CP_SCALE, EvaluationModel  # noqa: F401
from .registry import available_models, get_model_class, load_model, register_model  # noqa: F401
