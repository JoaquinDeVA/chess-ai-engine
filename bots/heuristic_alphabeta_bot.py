"""Heuristic alpha-beta bot: fixed-depth alpha-beta with a hand-crafted evaluation."""

from typing import Callable, Optional

import chess

from adversarial_search.chess_game_state import ChessGameState
from adversarial_search.chess_problem import ChessProblem
from bots._search_bots import AUTHOR, FixedDepthBot
from evaluation.heuristics import h3


class HeuristicAlphaBetaBot(FixedDepthBot):
    """Alpha-beta search to a fixed depth using a weighted-feature heuristic.

    :param eval_function: Heuristic ``(state, player) -> float`` (``h1``..``h4``); default ``h3``.
    :param max_depth: Plies to search.
    :param problem: Chess problem to search; a fresh one is created by default.
    :param name: Bot name (default ``HeuristicAlphaBetaBot``).
    """

    def __init__(
        self,
        eval_function: Callable[[ChessGameState, chess.Color], float] = h3,
        max_depth: int = 3,
        problem: Optional[ChessProblem] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or "HeuristicAlphaBetaBot", AUTHOR, max_depth, problem)
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

        :return: Bot information including the heuristic and depth in use.
        """
        info = super().get_info()
        info.update(
            description=f"Fixed-depth alpha-beta with heuristic {self.eval_function.__name__}",
            max_depth=self.max_depth,
        )
        return info
