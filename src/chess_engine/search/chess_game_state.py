"""
Chess game state implementation for adversarial search.

This module implements the GameState interface for chess using the python-chess library.
"""

from typing import Any, Optional

import chess

from .game_problems import GameState


class ChessGameState(GameState):
    """
    Chess implementation of GameState interface.

    Wraps a chess.Board object to provide the GameState interface
    required by adversarial search algorithms.
    """

    def __init__(self, board: Optional[chess.Board] = None):
        """
        Initialize chess game state.

        :param board: Chess board instance, creates new game if None
        :type board: Optional[chess.Board]
        """
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board.copy()

    def get_legal_moves(self) -> list[Any]:
        """
        Return a list of all legal moves from the current state.

        :return: List of legal chess moves
        :rtype: list[chess.Move]
        """
        return list(self.board.legal_moves)

    def make_move(self, move: Any) -> "ChessGameState":
        """
        Apply a move to the current state and return self.

        This method modifies the current state in place.

        :param move: The chess move to apply
        :type move: chess.Move
        :return: Self after applying the move
        :rtype: ChessGameState
        :raises ValueError: If the move is not legal
        """
        if move not in self.board.legal_moves:
            raise ValueError(f"Move {move} is not legal in current position")

        self.board.push(move)
        return self

    def is_over(self) -> bool:
        """
        Return True if the game is finished, False otherwise.

        Uses the same parameters as in engine/game_manager.py for consistency.

        :return: True if the game has ended, False if still in progress
        :rtype: bool
        """
        return self.board.is_game_over(claim_draw=True)

    def get_current_player(self) -> Any:
        """
        Return the identifier of the player whose turn it is.

        :return: chess.WHITE or chess.BLACK
        :rtype: chess.Color
        """
        return self.board.turn

    def get_winner(self) -> Optional[Any]:
        """
        Return the winner of the game if it's over, None otherwise.

        Returns winner format consistent with engine/game_manager.py.

        :return: "white", "black", None for draw, or None if game not over
        :rtype: Optional[str]
        """
        if not self.is_over():
            return None

        outcome = self.board.outcome(claim_draw=True)
        if outcome:
            if outcome.winner == chess.WHITE:
                return "white"
            elif outcome.winner == chess.BLACK:
                return "black"
            else:
                return None  # Draw
        else:
            return None

    def copy(self) -> "ChessGameState":
        """
        Return a deep copy of the game state.

        :return: A deep copy of this game state
        :rtype: ChessGameState
        """
        return ChessGameState(self.board)

    def __str__(self) -> str:
        """
        Return a string representation of the game state.

        :return: String representation of the chess board
        :rtype: str
        """
        return str(self.board)

    def get_board(self) -> chess.Board:
        """
        Get the underlying chess board.

        :return: The chess board instance
        :rtype: chess.Board
        """
        return self.board

    def get_fen(self) -> str:
        """
        Get the FEN string representation of the position.

        :return: FEN string of current position
        :rtype: str
        """
        return self.board.fen()

    def key(self) -> str:
        """
        Return the hashable identity of the state: its FEN string.

        :return: FEN of the position (the move counters are part of it).
        :rtype: str
        """
        return self.get_fen()
