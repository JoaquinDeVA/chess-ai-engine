"""
Chess game state implementation for adversarial search.

This module implements the GameState interface for chess using the python-chess library.
"""

from typing import Any, Hashable, Optional

import chess
from chess import Outcome, Termination

from .game_problems import GameState


class ChessGameState(GameState):
    """
    Chess implementation of GameState interface.

    Wraps a chess.Board object to provide the GameState interface
    required by adversarial search algorithms.

    The terminal test is the one of ``board.outcome(claim_draw=True)``, but the threefold
    repetition claim (slow in python-chess: it replays the game on every call) is computed from
    the transposition keys of the positions since the last irreversible move, which every state
    carries and extends in constant time per move.
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
        self._history, self._has_repeated = self._build_history(self.board)
        self._outcome: Optional[Outcome] = None
        self._outcome_known = False
        self._undo_stack: list[tuple] = []

    @staticmethod
    def _build_history(board: chess.Board) -> tuple[tuple, bool]:
        """
        Transposition keys of the positions since the last irreversible move.

        Mirrors the counting of ``Board.can_claim_threefold_repetition``: the current position and
        the ones before it, stopping at (and excluding the position before) an irreversible move.

        :param board: Board with its move stack
        :type board: chess.Board
        :return: The keys (oldest first, current last) and whether any key occurs twice or more
        :rtype: tuple[tuple, bool]
        """
        probe = board.copy()
        keys = [probe._transposition_key()]
        while probe.move_stack:
            move = probe.pop()
            if probe.is_irreversible(move):
                break
            keys.append(probe._transposition_key())
        keys.reverse()
        return tuple(keys), len(set(keys)) < len(keys)

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

        self.apply_move(move)
        self._undo_stack.pop()  # a move made with make_move is not meant to be undone
        return self

    def apply_move(self, move: Any) -> None:
        """
        Apply a legal move in place without checking it, keeping the history and caches updated.

        :param move: A legal chess move of this position
        :type move: chess.Move
        """
        self._undo_stack.append(
            (self._history, self._has_repeated, self._outcome, self._outcome_known)
        )
        irreversible = self.board.is_irreversible(move)
        self.board.push(move)
        key = self.board._transposition_key()
        if irreversible:
            self._history = (key,)
            self._has_repeated = False
        else:
            self._has_repeated = self._has_repeated or key in self._history
            self._history += (key,)
        self._outcome_known = False

    def undo_move(self) -> None:
        """Revert the last move applied with :meth:`apply_move`, restoring history and caches."""
        self.board.pop()
        self._history, self._has_repeated, self._outcome, self._outcome_known = (
            self._undo_stack.pop()
        )

    def _compute_outcome(self) -> Optional[Outcome]:
        """
        Outcome of ``board.outcome(claim_draw=True)`` without python-chess' slow threefold check.

        :return: The outcome, or None if the game is not over
        :rtype: Optional[chess.Outcome]
        """
        board = self.board
        outcome = board.outcome(claim_draw=False)
        if outcome is not None:
            return outcome
        if board.can_claim_fifty_moves():
            return Outcome(Termination.FIFTY_MOVES, None)
        if self._can_claim_threefold_repetition():
            return Outcome(Termination.THREEFOLD_REPETITION, None)
        return None

    def _can_claim_threefold_repetition(self) -> bool:
        """
        Same result as ``board.can_claim_threefold_repetition()``.

        The current position occurred three times, or some legal move reaches a position that
        already occurred twice (impossible unless some position has repeated).

        :return: True if a draw by threefold repetition can be claimed
        :rtype: bool
        """
        history = self._history
        if history.count(history[-1]) >= 3:
            return True
        if not self._has_repeated:
            return False
        board = self.board
        for move in board.legal_moves:
            board.push(move)
            try:
                if history.count(board._transposition_key()) >= 2:
                    return True
            finally:
                board.pop()
        return False

    def is_over(self) -> bool:
        """
        Return True if the game is finished, False otherwise.

        Uses the same parameters as in engine/game_manager.py for consistency.

        :return: True if the game has ended, False if still in progress
        :rtype: bool
        """
        if not self._outcome_known:
            self._outcome = self._compute_outcome()
            self._outcome_known = True
        return self._outcome is not None

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

        outcome = self._outcome
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
        clone = ChessGameState.__new__(ChessGameState)
        clone.board = self.board.copy()
        clone._history = self._history
        clone._has_repeated = self._has_repeated
        clone._outcome = self._outcome
        clone._outcome_known = self._outcome_known
        clone._undo_stack = []
        return clone

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

    def key(self) -> Hashable:
        """
        Return the hashable identity of the state.

        It identifies the same positions as the FEN string (pieces, turn, castling rights, legal
        en passant square, halfmove clock and fullmove number) without building the string.

        :return: Tuple of the board transposition key and the two move counters.
        :rtype: Hashable
        """
        board = self.board
        return (board._transposition_key(), board.halfmove_clock, board.fullmove_number)
