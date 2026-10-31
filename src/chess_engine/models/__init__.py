"""Trained position evaluations, loaded by name for the Rust core.

::

    from chess_engine.models import load_evaluator
    evaluator = load_evaluator("linear_pst_terms")

Defining, training and measuring the models is the laboratory's job (``chess_lab.models``).
"""

from .loader import (  # noqa: F401
    ARTIFACTS_DIR,
    artifact_path,
    h1_evaluator,
    installed_models,
    load_evaluator,
    load_h1_weights,
    split_spec,
)
