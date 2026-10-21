"""Tests of the estimators of the duration of the next iterative-deepening iteration."""

import pytest

from chess_engine.search import time_estimators as te


def test_ratio_squared_is_last_squared_over_previous():
    assert te.ratio_squared([(0.001, 10), (0.1, 100)]) == pytest.approx(10.0)


def test_ratio_squared_with_one_iteration_uses_the_default_growth():
    assert te.ratio_squared([(0.5, 10)]) == pytest.approx(0.5 * te.DEFAULT_GROWTH)


def test_ratio_squared_explodes_when_the_previous_iteration_was_nearly_instant():
    stalled = [(0.0005, 10), (0.5, 500)]
    assert te.ratio_squared(stalled) == pytest.approx(500.0)
    assert te.clamped_growth(stalled) == pytest.approx(0.5 * te.MAX_GROWTH)


@pytest.mark.parametrize(
    "estimator", [te.clamped_growth, te.median_growth, te.geometric_growth, te.node_growth]
)
def test_bounded_estimators_never_predict_more_than_max_growth_times_the_last(estimator):
    history = [(0.0001, 1), (0.2, 900), (50.0, 10**7)]
    assert estimator(history) <= history[-1][0] * te.MAX_GROWTH + 1e-12


@pytest.mark.parametrize(
    "estimator", [te.clamped_growth, te.median_growth, te.geometric_growth, te.node_growth]
)
def test_bounded_estimators_never_predict_less_than_min_growth_times_the_last(estimator):
    history = [(1.0, 100), (1.0, 100), (1.0, 100)]
    assert estimator(history) >= history[-1][0] * te.MIN_GROWTH - 1e-12


def test_median_growth_ignores_one_outlier():
    # growths 3, 5 and 100: the median is 5
    history = [(1.0, 1), (3.0, 1), (15.0, 1), (1500.0, 1)]
    assert te.median_growth(history) == pytest.approx(1500.0 * 5)


def test_geometric_growth_is_the_geometric_mean():
    history = [(1.0, 1), (2.0, 1), (8.0, 1)]  # growths 2 and 4
    assert te.geometric_growth(history) == pytest.approx(8.0 * 8**0.5)


def test_node_growth_follows_the_node_counts_not_the_clock():
    assert te.node_growth([(0.001, 100), (50.0, 600)]) == pytest.approx(50.0 * 6)


def test_fixed_factor_and_never_stop():
    assert te.fixed_factor([(2.0, 1)]) == pytest.approx(2.0 * te.DEFAULT_GROWTH)
    assert te.never_stop([(2.0, 1)]) == 0.0


def test_unknown_estimator_lists_the_valid_names():
    with pytest.raises(KeyError, match="available:.*clamped_growth"):
        te.get_estimator("nope")
    assert te.get_estimator("ratio_squared") is te.ratio_squared
