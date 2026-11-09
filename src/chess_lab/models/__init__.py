"""The models: definitions, saved files, features, targets and fitting.

The app does not load models: the window plays through the UCI binary, which embeds its own
(``chess_lab.scripts.export_uci_model``).
"""

from typing import Any, Dict, Iterable, Optional

import pandas as pd

from .definitions import NeuralEvalModel
from .fitting import neural
from .target import Representation


def new_model(model_class: type, rep: Optional[Representation] = None) -> NeuralEvalModel:
    """An untrained network: random weights, zero biases.

    :param model_class: A concrete model class, e.g. ``get_model_class("net_pst_w32")``.
    :param rep: Representation of the target; defaults to :class:`Representation`.
    :return: The model, with ``rep`` saved as a dictionary.
    """
    return neural.new_network_model(model_class, rep)


def fit_stream(
    model: NeuralEvalModel,
    batches: Iterable[pd.DataFrame],
    validation: pd.DataFrame,
    config: Dict[str, Any],
    cache_dir: Optional[Any] = None,
) -> Dict[str, Any]:
    """Train a network with Keras.

    :param model: The model to train; it keeps the best epoch.
    :param batches: Re-iterable frames with columns ``fen, cp, mate, white_to_move``.
    :param validation: Same columns, used for early stopping.
    :param config: Hyper-parameters; see :func:`chess_lab.models.fitting.neural.fit_stream`.
    :param cache_dir: Directory of the encoded frames, or None for no cache.
    :return: Summary of the run.
    """
    return neural.fit_stream(model, batches, validation, config, cache_dir)
