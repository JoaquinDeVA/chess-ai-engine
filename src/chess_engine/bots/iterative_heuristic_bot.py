"""Iterative-deepening alpha-beta with the hand-crafted heuristic ``h1``, transposition table and killers."""

from typing import Any, Optional

from chess_engine.bots._search_bots import AUTHOR, IterativeDeepeningBot
from chess_engine.models.native import h1_evaluator
from chess_engine.search.time_management import TimeManager


class IterativeHeuristicBot(IterativeDeepeningBot):
    """Clock-aware alpha-beta using the hand-crafted heuristic ``h1``.

    :param max_depth: Deepest iteration to run.
    :param time_manager: Time budgeting policy.
    :param search_options: Other options of :class:`IterativeDeepeningBot` (``quiescence_plies``,
        ``min_depth``, ``estimator``, ``max_tt_entries``, ``tablebase``).
    """

    def __init__(
        self,
        max_depth: int = 3,
        time_manager: Optional[TimeManager] = None,
        **search_options: Any,
    ) -> None:
        super().__init__(
            "IterativeHeuristicBot",
            AUTHOR,
            max_depth,
            time_manager,
            **search_options,
        )

    def native_evaluator(self):
        """The evaluation the core runs: ``h1``.

        :return: Evaluator for ``h1``.
        """
        return h1_evaluator()

    def get_info(self):
        """Describe the bot.

        :return: Bot information including the heuristic and depth cap in use.
        """
        info = super().get_info()
        info.update(
            description=(
                "Iterative-deepening alpha-beta with heuristic h1, "
                "transposition table, killer moves and MVV-LVA ordering"
            ),
            max_depth=self.max_depth,
        )
        return info
