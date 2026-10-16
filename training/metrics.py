"""Metrics that compare a model with Stockfish, in centipawns."""

from typing import Dict

import numpy as np

#: Evaluations within +-EQUAL_BAND cp count as "equal position" for the accuracy metric.
EQUAL_BAND = 50.0


def outcome_class(cp: np.ndarray, band: float = EQUAL_BAND) -> np.ndarray:
    """Classify evaluations as White better (+1), equal (0) or Black better (-1).

    :param cp: Evaluations in centipawns, White's point of view.
    :param band: Half-width of the "equal" zone.
    :return: Integer array of classes.
    """
    cp = np.asarray(cp)
    return np.where(cp > band, 1, np.where(cp < -band, -1, 0))


def compute_metrics(predicted_cp: np.ndarray, true_cp: np.ndarray) -> Dict[str, float]:
    """Compare predictions with Stockfish evaluations.

    * ``mae_cp`` / ``rmse_cp``: absolute and root-mean-square error in centipawns.
    * ``accuracy``: share of positions where model and Stockfish agree on who is better, with
      ``+-50`` cp counted as equal (**the headline "% of agreement" of the project**).
    * ``within_50`` / ``within_100``: share of predictions within 50 / 100 cp of Stockfish.

    :param predicted_cp: Model evaluations in centipawns.
    :param true_cp: Stockfish evaluations in centipawns.
    :return: Dictionary of metrics (shares are in ``[0, 1]``).
    :raises ValueError: If the arrays are empty or have different lengths.
    """
    predicted = np.asarray(predicted_cp, dtype=np.float64)
    truth = np.asarray(true_cp, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.size == 0:
        raise ValueError("predicted_cp and true_cp must be non-empty and have the same shape")

    error = predicted - truth
    return {
        "n_test": int(truth.size),
        "mae_cp": float(np.abs(error).mean()),
        "rmse_cp": float(np.sqrt((error**2).mean())),
        "accuracy": float((outcome_class(predicted) == outcome_class(truth)).mean()),
        "within_50": float((np.abs(error) <= 50).mean()),
        "within_100": float((np.abs(error) <= 100).mean()),
    }
