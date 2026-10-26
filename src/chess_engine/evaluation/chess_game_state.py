"""The chess position the evaluation functions receive."""

import chess


class ChessGameState:
    """A chess position handed to an evaluation function ``(state, player) -> float``.

    It wraps a copy of a board. The search itself runs in the Rust core; this class only keeps the
    signature of the heuristics and of ``ModelEvaluator`` (and of the callbacks the core makes to
    Python for evaluations it cannot run natively).

    :param board: Position to wrap; the standard starting position by default.
    """

    def __init__(self, board: chess.Board | None = None):
        self.board = chess.Board() if board is None else board.copy()

    def get_board(self) -> chess.Board:
        """The wrapped board.

        :return: The chess board.
        """
        return self.board

    def get_current_player(self) -> chess.Color:
        """The side to move.

        :return: ``chess.WHITE`` or ``chess.BLACK``.
        """
        return self.board.turn

    def copy(self) -> "ChessGameState":
        """An independent copy of this state.

        :return: A new state with its own board.
        """
        return ChessGameState(self.board)

    def __str__(self) -> str:
        """The board as text.

        :return: The board drawn with letters.
        """
        return str(self.board)
