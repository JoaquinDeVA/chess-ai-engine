"""Conversion of Stockfish evaluations into normalized training targets.

Decisions (they define what "accuracy against Stockfish" means in ``MODELS_HISTORY.md``):

* A centipawn evaluation is clipped to ``+-CP_SCALE`` (1267 cp, the 0.5% quantile of the course
  dataset, so ~99% of the positions are untouched) and divided by ``CP_SCALE``: ``y`` in
  ``[-1, 1]``.
* A forced mate within ``max_mate_distance`` moves becomes ``y = +-1`` (the sign tells who mates).
* Longer mates are dropped: they are rare, and their evaluation says little about the position
  itself.
* Evaluations are always from **White's point of view** (positive = White is better).
"""

from typing import Optional

import numpy as np

from models.base import CP_SCALE

#: Largest ``|mate_distance|`` that is kept as a training example.
MAX_MATE_DISTANCE = 2


def cp_to_target(cp: np.ndarray) -> np.ndarray:
    """Normalize centipawns to ``[-1, 1]``.

    :param cp: Evaluation(s) in centipawns, White's point of view.
    :return: ``clip(cp / CP_SCALE, -1, 1)``.
    """
    return np.clip(np.asarray(cp, dtype=np.float64) / CP_SCALE, -1.0, 1.0)


def mate_to_target(
    mate_distance: np.ndarray, max_mate_distance: int = MAX_MATE_DISTANCE
) -> np.ndarray:
    """Map mate distances to ``+-1``, or ``NaN`` when the mate is too far away.

    :param mate_distance: Moves to mate; positive when White mates, negative when Black does.
    :param max_mate_distance: Largest ``|mate_distance|`` that is kept.
    :return: ``sign(mate_distance)``, with ``NaN`` where ``|mate_distance| > max_mate_distance``.
    """
    distance = np.asarray(mate_distance, dtype=np.float64)
    return np.where(np.abs(distance) <= max_mate_distance, np.sign(distance), np.nan)


def row_targets(
    is_mate: np.ndarray,
    cp: np.ndarray,
    mate_distance: np.ndarray,
    max_mate_distance: Optional[int] = MAX_MATE_DISTANCE,
) -> np.ndarray:
    """Compute the target of every row of an evaluations table.

    :param is_mate: Boolean array, True where the evaluation is a forced mate.
    :param cp: Centipawn column (ignored on mate rows, may be NaN).
    :param mate_distance: Mate-distance column (ignored on non-mate rows, may be NaN).
    :param max_mate_distance: Largest mate distance kept; ``None`` uses :data:`MAX_MATE_DISTANCE`.
    :return: Targets in ``[-1, 1]``; ``NaN`` marks rows that must be dropped.
    """
    limit = MAX_MATE_DISTANCE if max_mate_distance is None else max_mate_distance
    is_mate = np.asarray(is_mate, dtype=bool)
    with np.errstate(invalid="ignore"):
        return np.where(is_mate, mate_to_target(mate_distance, limit), cp_to_target(cp))
