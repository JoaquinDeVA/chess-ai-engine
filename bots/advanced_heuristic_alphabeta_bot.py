"""Advanced heuristic alpha-beta bot: iterative deepening, transposition table, killer moves."""

from typing import Callable, Optional

import chess

from adversarial_search.chess_game_state import ChessGameState
from adversarial_search.chess_problem import ChessProblem
from adversarial_search.time_management import TimeManager
from bots._search_bots import AUTHOR, IterativeDeepeningBot
from evaluation.heuristics import h3


class AdvancedHeuristicAlphaBetaBot(IterativeDeepeningBot):
    """Clock-aware alpha-beta using a hand-crafted heuristic.

    :param eval_function: Heuristic ``(state, player) -> float`` (``h1``..``h4``); default ``h3``.
    :param max_depth: Deepest iteration to run.
    :param problem: Chess problem to search; a fresh one is created by default.
    :param time_manager: Time budgeting policy.
    """

    def __init__(
        self,
        eval_function: Callable[[ChessGameState, chess.Color], float] = h3,
        max_depth: int = 3,
        problem: Optional[ChessProblem] = None,
        time_manager: Optional[TimeManager] = None,
    ) -> None:
        super().__init__(
            "AdvancedHeuristicAlphaBetaBot", AUTHOR, max_depth, time_manager, problem=problem
        )
        self.eval_function = eval_function

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
