"""Per-move time budgeting shared by the time-aware bots."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeManager:
    """Decides how long a bot may think about a move.

    The budget for a move is a fraction of the time left on the clock, clamped to
    ``[min_turn_time, max_turn_time]`` and never more than ``max_clock_share`` of the clock.
    A search may run past its budget by ``extension`` (to finish a depth it has started) minus a
    ``safety_margin`` that pays for the overhead of unwinding the search.

    :param move_fraction: Share of the remaining clock spent on one move.
    :param min_turn_time: Lower bound of the budget, in seconds.
    :param max_turn_time: Upper bound of the budget, in seconds.
    :param max_clock_share: Never plan to spend more than this share of the remaining clock.
    :param extension: Multiplier applied to the budget to obtain the hard limit.
    :param safety_margin: Seconds subtracted from the hard limit.
    """

    move_fraction: float = 0.15
    min_turn_time: float = 1.5
    max_turn_time: float = 40.0
    max_clock_share: float = 0.5
    extension: float = 1.5
    safety_margin: float = 0.3

    def turn_budget(self, time_left: float) -> float:
        """Soft time budget for the next move.

        :param time_left: Seconds left on the player's clock (may be ``inf`` without a clock).
        :return: Budget in seconds.
        """
        budget = max(self.min_turn_time, min(self.max_turn_time, time_left * self.move_fraction))
        return min(budget, time_left * self.max_clock_share)

    def hard_limit(self, budget: float) -> float:
        """Elapsed time after which a running search must stop.

        :param budget: Soft budget returned by :meth:`turn_budget`.
        :return: Hard limit in seconds.
        """
        return budget * self.extension - self.safety_margin
