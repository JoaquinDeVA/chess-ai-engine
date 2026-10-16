"""Non-tactical alpha-beta bot: the control group for :class:`TacticalAlphaBetaBot`."""

from typing import Callable

import chess

from adversarial_search.chess_game_state import ChessGameState
from bots.heuristic_alphabeta_bot import HeuristicAlphaBetaBot
from evaluation.heuristics import h3


class NonTacticalAlphaBetaBot(HeuristicAlphaBetaBot):
    """Fixed-depth alpha-beta without Prolog move ordering.

    It searches exactly like :class:`bots.tactical_alphabeta_bot.TacticalAlphaBetaBot` except
    that moves are not reordered by tactical patterns, which makes it the baseline to measure
    what the tactical knowledge buys (fewer visited nodes at equal depth).

    :param eval_function: Heuristic ``(state, player) -> float``; default ``h3``.
    :param max_depth: Plies to search.
    :param name: Bot name.
    """

    def __init__(
        self,
        eval_function: Callable[[ChessGameState, chess.Color], float] = h3,
        max_depth: int = 3,
        name: str = "NonTacticalAlphaBetaBot",
    ):
        super().__init__(eval_function, max_depth, name=name)
