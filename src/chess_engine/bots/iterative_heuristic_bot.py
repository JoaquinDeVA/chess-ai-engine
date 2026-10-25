"""Iterative-deepening alpha-beta with a hand-crafted heuristic, transposition table and killers."""

from typing import Any, Callable, Optional

import chess

from chess_engine.bots._search_bots import AUTHOR, IterativeDeepeningBot
from chess_engine.evaluation.heuristics import h1
from chess_engine.models.native import heuristic_evaluator
from chess_engine.search.chess_game_state import ChessGameState
from chess_engine.search.time_management import TimeManager


class IterativeHeuristicBot(IterativeDeepeningBot):
    """Clock-aware alpha-beta using a hand-crafted heuristic.

    :param eval_function: Heuristic ``(state, player) -> float`` (e.g. ``h1``); default ``h1``.
    :param max_depth: Deepest iteration to run.
    :param time_manager: Time budgeting policy.
    :param search_options: Other options of :class:`IterativeDeepeningBot` (``quiescence_plies``,
        ``min_depth``, ``estimator``, ``max_tt_entries``, ``tablebase``).
    """

    def __init__(
        self,
        eval_function: Callable[[ChessGameState, chess.Color], float] = h1,
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
        self.eval_function = eval_function

    def native_evaluator(self):
        """The evaluation the core runs: ``h1`` natively, any other heuristic by callback.

        :return: Evaluator for ``eval_function``.
        """
        return heuristic_evaluator(self.eval_function)

    def evaluate(self, state: ChessGameState, player: chess.Color) -> float:
        """Evaluate with the configured heuristic.

        :param state: Position to evaluate.
        :param player: Player the value is for.
        :return: Expected score for ``player`` in ``[0.01, 0.99]``.
        """
        return self.eval_function(state, player)

    def get_info(self):
        """Describe the bot.

        :return: Bot information including the heuristic and depth cap in use.
        """
        info = super().get_info()
        info.update(
            description=(
                f"Iterative-deepening alpha-beta with heuristic {self.eval_function.__name__}, "
                "transposition table, killer moves and MVV-LVA ordering"
            ),
            max_depth=self.max_depth,
        )
        return info
