"""Estimators of the duration of the next iteration of an iterative-deepening search.

Iterative deepening searches depth 1, 2, 3 ... and needs to know, before starting depth ``d + 1``,
whether it will finish in the time that is left. Each estimator receives the finished iterations
as ``(seconds, nodes)`` pairs, oldest first, and returns the predicted seconds of the next one.
They are pure functions, so the same code decides in the search and in the offline comparison
(``scripts/bench/compare_time_estimators.py``).
"""

import math
import statistics
from typing import Callable, Dict, Sequence, Tuple

#: A finished iteration: ``(seconds, nodes)``.
Iteration = Tuple[float, int]
Estimator = Callable[[Sequence[Iteration]], float]

#: Durations below this (seconds) are too short to give a meaningful ratio.
MIN_MEASURABLE = 1e-6
#: Bounds of the growth between two consecutive iterations. Alpha-beta with a decent move order
#: grows the cost of an iteration by 2 to 8 times per ply; a ratio outside this band is noise.
MIN_GROWTH, MAX_GROWTH = 2.0, 8.0
#: Growth assumed when there is nothing to measure it from.
DEFAULT_GROWTH = 4.0


def _clamp(value: float, low: float = MIN_GROWTH, high: float = MAX_GROWTH) -> float:
    return max(low, min(high, value))


def _growths(history: Sequence[Iteration]) -> list:
    """Ratios of consecutive durations that are long enough to measure."""
    return [
        later[0] / earlier[0]
        for earlier, later in zip(history, history[1:], strict=False)
        if earlier[0] > MIN_MEASURABLE and later[0] > MIN_MEASURABLE
    ]


def ratio_squared(history: Sequence[Iteration]) -> float:
    """Last duration times its growth over the one before (the original rule, unbounded).

    The prediction is ``last ** 2 / previous``: one stall in ``last`` or a near-instant
    ``previous`` multiplies it by hundreds.
    """
    if len(history) < 2:
        return history[-1][0] * DEFAULT_GROWTH
    last, previous = history[-1][0], history[-2][0]
    growth = last / previous if previous > MIN_MEASURABLE and last > MIN_MEASURABLE else 1.5
    return last * growth


def fixed_factor(history: Sequence[Iteration]) -> float:
    """Last duration times a constant growth (``DEFAULT_GROWTH``)."""
    return history[-1][0] * DEFAULT_GROWTH


def clamped_growth(history: Sequence[Iteration]) -> float:
    """Last duration times its growth over the previous one, kept between 2 and 8."""
    growths = _growths(history[-2:])
    return history[-1][0] * (_clamp(growths[0]) if growths else DEFAULT_GROWTH)


def median_growth(history: Sequence[Iteration]) -> float:
    """Last duration times the median of the last three growths, kept between 2 and 8."""
    growths = _growths(history[-4:])
    return history[-1][0] * (_clamp(statistics.median(growths)) if growths else DEFAULT_GROWTH)


def geometric_growth(history: Sequence[Iteration]) -> float:
    """Last duration times the geometric mean of all the measurable growths, kept between 2 and 8."""
    growths = _growths(history)
    if not growths:
        return history[-1][0] * DEFAULT_GROWTH
    return history[-1][0] * _clamp(math.exp(statistics.fmean(math.log(g) for g in growths)))


def node_growth(history: Sequence[Iteration]) -> float:
    """Last duration times the growth of the node count, kept between 2 and 8.

    Node counts do not suffer from a stalled clock, so the ratio is stable; it assumes that the
    time per node does not change from one iteration to the next.
    """
    if len(history) < 2 or history[-2][1] <= 0 or history[-1][1] <= 0:
        return history[-1][0] * DEFAULT_GROWTH
    return history[-1][0] * _clamp(history[-1][1] / history[-2][1])


def never_stop(history: Sequence[Iteration]) -> float:
    """Predict zero seconds, so the estimate never stops the deepening (measurement tool)."""
    return 0.0


#: Estimators by name.
ESTIMATORS: Dict[str, Estimator] = {
    "never_stop": never_stop,
    "ratio_squared": ratio_squared,
    "fixed_factor": fixed_factor,
    "clamped_growth": clamped_growth,
    "median_growth": median_growth,
    "geometric_growth": geometric_growth,
    "node_growth": node_growth,
}


def get_estimator(name: str) -> Estimator:
    """Estimator registered under ``name``.

    :param name: One of :data:`ESTIMATORS`.
    :return: The estimator.
    :raises KeyError: If the name is unknown (the message lists the valid ones).
    """
    try:
        return ESTIMATORS[name]
    except KeyError:
        raise KeyError(f"unknown estimator {name!r}; available: {sorted(ESTIMATORS)}") from None
