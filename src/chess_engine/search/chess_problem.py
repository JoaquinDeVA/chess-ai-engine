"""
Chess problem implementation for adversarial search.

This module implements the GameProblem interface for chess using ChessGameState.
"""

from typing import Any, List

import chess

from .chess_game_state import ChessGameState
from .game_problems import GameProblem, GameState


def chess_noisy_moves(state: GameState) -> List[chess.Move]:
    """Legal captures and promotions, the moves a quiescence search follows.

    :param state: Chess position.
    :return: Legal moves that capture or promote.
    """
    board = state.get_board()
    return [m for m in board.legal_moves if m.promotion or board.is_capture(m)]


class ChessProblem(GameProblem):
    """
    Chess implementation of GameProblem interface.

    Provides chess-specific game logic and utility evaluation
    for adversarial search algorithms.
    """

    def __init__(self, initial_board: chess.Board = None):
        """
        Initialize chess problem.

        :param initial_board: Starting chess position, creates new game if None
        :type initial_board: Optional[chess.Board]
        """
        if initial_board is None:
            self.initial = ChessGameState()
        else:
            self.initial = ChessGameState(initial_board)

    def utility(self, state: GameState, player: Any) -> float:
        """
        Return the value of this final state to player.

        For terminal states, returns standard game values.
        This method should only be called on terminal states.

        :param state: The terminal game state
        :type state: GameState
        :param player: The player for whom to calculate utility (chess.WHITE or chess.BLACK)
        :type player: chess.Color
        :return: 1.0 for win, 0.5 for draw, 0.0 for loss
        :rtype: float
        :raises ValueError: If state is not ChessGameState or not terminal
        """
        if not isinstance(state, GameState) or not state.is_over():
            raise ValueError("state must be ChessGameState and terminal.")

        if player not in (chess.WHITE, chess.BLACK):
            raise ValueError("player must be chess.WHITE or chess.BLACK.")

        winner = state.get_winner()  # "white", "black" or None for a draw
        if winner is None:
            return 0.5
        winner_color = chess.WHITE if winner == "white" else chess.BLACK
        return 1.0 if winner_color == player else 0.0
