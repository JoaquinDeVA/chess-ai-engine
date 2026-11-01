"""The models: definitions, saved files, features, targets and fitting.

The app does not load models: the window plays through the UCI binary, which embeds its own
(``chess_lab.scripts.export_uci_model``).
"""

from typing import Any, Dict, Iterable, Optional

import pandas as pd

from . import linear_eval, neural_eval
from .definitions import NeuralEvalModel, SparseModel
from .representation import Representation


def new_model(model_class: type, rep: Optional[Representation] = None) -> SparseModel:
    """An untrained model of a class: zero weights for a linear one, random for a network.

    :param model_class: A concrete model class, e.g. ``get_model_class("linear_pst")``.
    :param rep: Representation of the target; defaults to :class:`Representation`.
    :return: The model, with ``rep`` saved as a dictionary.
    """
    if issubclass(model_class, NeuralEvalModel):
        return neural_eval.new_network_model(model_class, rep)
    return linear_eval.new_linear_model(model_class, rep)


def fit_stream(
    model: SparseModel,
    batches: Iterable[pd.DataFrame],
    validation: pd.DataFrame,
    config: Dict[str, Any],
    cache_dir: Optional[Any] = None,
) -> Dict[str, Any]:
    """Train a model (Adam for a linear one, Keras for a network).

    :param model: The model to train; it keeps the best epoch.
    :param batches: Re-iterable frames with columns ``fen, cp, mate, white_to_move``.
    :param validation: Same columns, used for early stopping.
    :param config: Hyper-parameters; see :func:`chess_lab.models.linear_eval.fit_stream`.
    :param cache_dir: Directory of the encoded frames, or None for no cache.
    :return: Summary of the run.
    """
    module = neural_eval if isinstance(model, NeuralEvalModel) else linear_eval
    return module.fit_stream(model, batches, validation, config, cache_dir)
