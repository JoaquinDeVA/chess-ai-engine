"""Tests of src/chess_lab/scripts/bench/compare_quiescence.py (statistics and arguments only; no search)."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def script():
    path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "chess_lab"
        / "scripts"
        / "bench"
        / "compare_quiescence.py"
    )
    spec = importlib.util.spec_from_file_location("compare_quiescence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_difference_of_identical_outcomes_is_zero(script):
    assert script.paired_difference([1, 0, 1, 1], [1, 0, 1, 1]) == (0.0, 0.0)


def test_paired_difference_mean_and_standard_error(script):
    mean, error = script.paired_difference([1, 1, 1, 0], [0, 1, 0, 0])
    assert mean == pytest.approx(0.5)
    # differences 1, 0, 1, 0: variance 1/3 over 4 positions
    assert error == pytest.approx((1 / 3 / 4) ** 0.5)


def test_paired_difference_needs_the_same_positions(script):
    with pytest.raises(ValueError, match="same positions"):
        script.paired_difference([1, 0], [1])


def test_paired_difference_with_fewer_than_two_positions_has_no_error(script):
    assert script.paired_difference([], []) == (0.0, 0.0)
    assert script.paired_difference([1], [0]) == (1.0, 0.0)


def test_arguments_default_to_the_reference_players(script):
    args = script.parse_args([])
    assert args.players == "h1,linear_pst_terms"
    assert args.positions == 1000
