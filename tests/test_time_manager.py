import math

from adversarial_search.time_management import TimeManager


def test_budget_is_a_fraction_of_the_clock_within_bounds():
    tm = TimeManager()
    assert tm.turn_budget(100.0) == 15.0
    assert tm.turn_budget(1000.0) == tm.max_turn_time
    assert tm.turn_budget(5.0) == 1.5


def test_budget_never_exceeds_half_the_remaining_clock():
    tm = TimeManager()
    assert tm.turn_budget(2.0) == 1.0
    assert tm.turn_budget(0.2) == 0.1


def test_no_clock_gives_the_maximum_budget():
    assert TimeManager().turn_budget(math.inf) == 40.0


def test_hard_limit_extends_the_budget_minus_a_safety_margin():
    tm = TimeManager()
    assert math.isclose(tm.hard_limit(10.0), 14.7)
    assert not tm.expired(10.0, 14.6)
    assert tm.expired(10.0, 14.7)
