"""The estimators of the core, reached through ``chess_engine._core.estimate`` (the rules themselves
are tested in ``rust/src/search/estimators.rs``)."""

import pytest

from chess_engine import _core


def estimate(name, history):
    return _core.estimate(name, history)


def test_ratio_squared_is_last_squared_over_previous():
    assert estimate("ratio_squared", [(0.001, 10), (0.1, 100)]) == pytest.approx(10.0)
    assert estimate("ratio_squared", [(0.5, 10)]) == pytest.approx(2.0)


def test_ratio_squared_explodes_when_the_previous_iteration_was_nearly_instant():
    stalled = [(0.0005, 10), (0.5, 500)]
    assert estimate("ratio_squared", stalled) == pytest.approx(500.0)
    assert estimate("clamped_growth", stalled) == pytest.approx(0.5 * 8.0)


@pytest.mark.parametrize(
    "name", ["clamped_growth", "median_growth", "geometric_growth", "node_growth"]
)
def test_bounded_estimators_stay_between_double_and_eight_times_the_last(name):
    for history in ([(0.0001, 1), (0.2, 900), (50.0, 10**7)], [(1.0, 100), (1.0, 100), (1.0, 100)]):
        last = history[-1][0]
        assert 2.0 * last - 1e-12 <= estimate(name, history) <= 8.0 * last + 1e-12


def test_geometric_growth_is_the_geometric_mean():
    assert estimate("geometric_growth", [(1.0, 1), (2.0, 1), (8.0, 1)]) == pytest.approx(
        8.0 * 8**0.5
    )


def test_fixed_factor_and_never_stop():
    assert estimate("fixed_factor", [(2.0, 1)]) == pytest.approx(8.0)
    assert estimate("never_stop", [(2.0, 1)]) == 0.0


def test_unknown_estimator_lists_the_valid_names():
    with pytest.raises(ValueError, match="available:.*clamped_growth"):
        estimate("nope", [(1.0, 1)])
    assert _core.DEFAULT_ESTIMATOR in _core.ESTIMATOR_NAMES
