"""Tests of scripts/bench/compare_time_estimators.py on synthetic series."""

import importlib.util
from pathlib import Path

import pytest

from chess_engine.search import time_estimators as te


@pytest.fixture(scope="module")
def script():
    path = (
        Path(__file__).resolve().parent.parent / "scripts" / "bench" / "compare_time_estimators.py"
    )
    spec = importlib.util.spec_from_file_location("compare_time_estimators", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def series(*seconds, truncated_last=False):
    """Iterations of the given durations with 1000 nodes each."""
    last = len(seconds)
    return [(i + 1, s, 1000, truncated_last and i + 1 == last) for i, s in enumerate(seconds)]


def test_hard_limit_matches_the_time_manager(script):
    assert script.hard_limit(40.0) == pytest.approx(59.7)


def test_oracle_depth_counts_iterations_that_fit_the_budget_with_overshoot(script):
    s = series(1, 4, 16, 64)  # cumulative 1, 5, 21, 85
    assert script.oracle_depth(s, budget=20.0) == 3  # 21 <= 23
    assert script.oracle_depth(s, budget=10.0) == 2


def test_simulate_stops_when_the_estimate_does_not_fit(script):
    s = series(0.01, 0.04, 0.2, 1.0, 5.0, 25.0)
    out = script.simulate(s, te.fixed_factor, budget=3.0)
    # after depth 4 (cumulative 1.25 s) the next iteration is predicted at 4 s > 1.75 s * 1.15
    assert out["depth"] == 4 and not out["cut"]


def test_simulate_marks_an_iteration_cut_by_the_hard_limit(script):
    s = series(0.01, 0.04, 0.2, 100.0, truncated_last=True)
    out = script.simulate(s, te.never_stop, budget=2.0)
    assert (
        out["depth"] == 3 and out["cut"] and out["seconds"] == pytest.approx(script.hard_limit(2.0))
    )


def test_simulate_flags_a_premature_stop_caused_by_a_stalled_iteration(script):
    # depth 3 stalled (0.9 s after a 0.01 s depth 2); depth 4 and 5 are cheap again
    s = series(0.001, 0.01, 0.9, 0.3, 0.5)
    original = script.simulate(s, te.ratio_squared, budget=10.0)
    robust = script.simulate(s, te.clamped_growth, budget=10.0)
    assert original["premature"] and original["depth"] == 3
    assert robust["depth"] == 5 and not robust["premature"]


def test_evaluate_averages_and_rejects_empty_input(script):
    result = script.evaluate([series(0.01, 0.04, 0.2, 1.0)], te.never_stop, budget=60.0)
    assert result["depth"] == 4 and result["cut"] == 0.0
    with pytest.raises(ValueError, match="no series"):
        script.evaluate([], te.never_stop, 10.0)
