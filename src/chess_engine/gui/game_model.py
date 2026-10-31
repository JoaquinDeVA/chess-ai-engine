"""The game on screen: board, clocks, undo and the ways a game ends. No Qt in here."""

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import chess
import chess.pgn

#: Pawn units of each piece, for the material count shown beside the board.
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


@dataclass(frozen=True)
class Outcome:
    """How a game ended.

    :param score: ``"1-0"``, ``"0-1"`` or ``"1/2-1/2"``.
    :param reason: Short text for the end-of-game banner, e.g. ``"checkmate"``.
    """

    score: str
    reason: str

    @property
    def winner(self) -> Optional[chess.Color]:
        """Colour that won, or None for a draw."""
        return {"1-0": chess.WHITE, "0-1": chess.BLACK}.get(self.score)


class Clock:
    """Two chess clocks with an increment; only the side to move runs.

    :param base: Seconds each side starts with.
    :param increment: Seconds added to a side after it moves.
    :param now: Source of the time, replaceable in tests.
    """

    def __init__(
        self, base: float, increment: float = 0.0, now: Callable[[], float] = time.monotonic
    ):
        self.increment = increment
        self._now = now
        self._left = {chess.WHITE: float(base), chess.BLACK: float(base)}
        self._running: Optional[chess.Color] = None
        self._since = 0.0

    def remaining(self, color: chess.Color) -> float:
        """Seconds left for a side, counting the time its turn has run.

        :param color: Side to ask for.
        :return: Seconds left, never below zero.
        """
        left = self._left[color]
        if self._running == color:
            left -= self._now() - self._since
        return max(0.0, left)

    def start(self, color: chess.Color) -> None:
        """Run the clock of ``color`` (stopping the other one).

        :param color: Side whose time starts to run.
        """
        self.stop()
        self._running = color
        self._since = self._now()

    def stop(self) -> None:
        """Stop the running clock, charging the time that passed."""
        if self._running is not None:
            self._left[self._running] = self.remaining(self._running)
            self._running = None

    def press(self, color: chess.Color) -> None:
        """End the turn of ``color``: stop its clock, add the increment, start the other one.

        :param color: Side that has just moved.
        """
        self.stop()
        self._left[color] += self.increment
        self.start(not color)

    def snapshot(self) -> Tuple[float, float]:
        """Times left as ``(white, black)``, to restore them after an undo."""
        return self.remaining(chess.WHITE), self.remaining(chess.BLACK)

    def restore(self, snapshot: Tuple[float, float], running: Optional[chess.Color]) -> None:
        """Set the times and run the clock of ``running`` (None leaves both stopped).

        :param snapshot: Value returned by :meth:`snapshot`.
        :param running: Side whose clock runs afterwards.
        """
        self._running = None
        self._left = {chess.WHITE: snapshot[0], chess.BLACK: snapshot[1]}
        if running is not None:
            self.start(running)

    def flagged(self) -> Optional[chess.Color]:
        """Side whose time has run out, if any."""
        for color in (chess.WHITE, chess.BLACK):
            if self.remaining(color) <= 0.0:
                return color
        return None


def material(board: chess.Board, color: chess.Color) -> int:
    """Material of a side in pawns (no king).

    :param board: Position.
    :param color: Side to count.
    :return: Sum of the values of its pieces.
    """
    return sum(
        PIECE_VALUES[piece.piece_type]
        for piece in board.piece_map().values()
        if piece.color == color
    )


def ai_accepts_draw(board: chess.Board, ai_color: chess.Color) -> bool:
    """Whether the AI takes a draw offer: when it is not ahead in material.

    :param board: Position when the offer is made.
    :param ai_color: Side the AI plays.
    :return: True if the AI has no more material than the opponent.
    """
    return material(board, ai_color) <= material(board, not ai_color)


class GameModel:
    """State of one game: position, SAN list, clocks and result.

    :param fen: Starting position (default: the initial one).
    :param clock: Clocks, or None for a game without time control.
    """

    def __init__(self, fen: Optional[str] = None, clock: Optional[Clock] = None):
        self.board = chess.Board(fen) if fen else chess.Board()
        self.clock = clock
        self.san_moves: List[str] = []
        self._times: List[Tuple[float, float]] = []
        self._decided: Optional[Outcome] = None
        if self.clock is not None:
            self.clock.start(self.board.turn)

    @property
    def turn(self) -> chess.Color:
        """Side to move."""
        return self.board.turn

    def legal_moves_from(self, square: chess.Square) -> List[chess.Move]:
        """Legal moves of the piece on ``square`` (four per target when it promotes).

        :param square: Origin square.
        :return: Legal moves from it; empty if the game is over or the square has none.
        """
        if self.outcome() is not None:
            return []
        return list(self.board.generate_legal_moves(from_mask=chess.BB_SQUARES[square]))

    def make_move(self, move: chess.Move) -> str:
        """Play a move and switch the clocks.

        :param move: Move to play.
        :return: Its SAN.
        :raises ValueError: If the game is over or the move is not legal.
        """
        if self.outcome() is not None:
            raise ValueError("the game is over")
        if move not in self.board.legal_moves:
            raise ValueError(f"illegal move {move.uci()}")
        mover = self.board.turn
        times = self.clock.snapshot() if self.clock is not None else (0.0, 0.0)
        san = self.board.san(move)
        self.board.push(move)
        self._times.append(times)
        self.san_moves.append(san)
        if self.clock is not None:
            self.clock.press(mover)
        return san

    def undo(self, plies: int = 1) -> int:
        """Take back moves and give both clocks the time they had.

        :param plies: Half-moves to take back (limited to the moves played).
        :return: How many were taken back.
        """
        plies = min(plies, len(self.board.move_stack))
        if plies <= 0 or self._decided is not None:
            return 0
        for _ in range(plies):
            self.board.pop()
            self.san_moves.pop()
            snapshot = self._times.pop()
        if self.clock is not None:
            self.clock.restore(snapshot, self.board.turn)
        return plies

    def resign(self, color: chess.Color) -> None:
        """End the game with a win for the other side.

        :param color: Side that resigns.
        """
        self._finish(Outcome("0-1" if color == chess.WHITE else "1-0", "resignation"))

    def agree_draw(self) -> None:
        """End the game as a draw by agreement."""
        self._finish(Outcome("1/2-1/2", "draw by agreement"))

    def can_claim_draw(self) -> bool:
        """Whether the side to move may claim a draw (repetition or fifty moves)."""
        return self.outcome() is None and self.board.can_claim_draw()

    def outcome(self) -> Optional[Outcome]:
        """The result if the game has ended, else None.

        Checkmate, stalemate and insufficient material end it by themselves; a threefold repetition
        or the fifty-move rule also do, as on chess.com. A clock that reaches zero loses, unless the
        other side cannot mate, which is a draw. Those that depend on the position are recomputed, so
        taking a move back reopens the game; resignation, agreed draw and timeout are final.
        """
        if self._decided is not None:
            return self._decided
        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None:
            if self.clock is not None:
                self.clock.stop()
            return Outcome(outcome.result(), outcome.termination.name.lower().replace("_", " "))
        if self.clock is not None:
            flagged = self.clock.flagged()
            if flagged is not None:
                if self.board.has_insufficient_material(not flagged):
                    return self._finish(Outcome("1/2-1/2", "timeout vs insufficient material"))
                return self._finish(Outcome("0-1" if flagged == chess.WHITE else "1-0", "timeout"))
        return None

    def _finish(self, outcome: Outcome) -> Outcome:
        """Record a final result and stop the clocks."""
        self._decided = outcome
        if self.clock is not None:
            self.clock.stop()
        return outcome

    def pgn(self, white: str, black: str) -> str:
        """The game in PGN.

        :param white: Name of the White player.
        :param black: Name of the Black player.
        :return: PGN text with the result when the game has ended.
        """
        game = chess.pgn.Game.from_board(self.board)
        game.headers.update({"Event": "ChessAIEngine game", "White": white, "Black": black})
        result = self.outcome()
        game.headers["Result"] = result.score if result else "*"
        return str(game)
