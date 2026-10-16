"""Tactical-pattern detection with the Prolog knowledge base ``knowledge/chess_patterns.pl``.

The knowledge base recognizes pins, forks, skewers, captures and promotions;
:class:`PrologPatternDetector` loads a position as facts and asks which legal moves *create*
each pattern.

This module needs SWI-Prolog and the ``pyswip`` package (``pip install -e '.[prolog]'``); both
are optional for the rest of the engine.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import chess

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE = Path(__file__).resolve().parent / "chess_patterns.pl"

# (Prolog predicate, key in the result dictionary); all of them bind FromR, FromC, ToR, ToC, Score.
_SCORED_PATTERNS = (
    ("move_creates_absolute_pin", "absolute_pin_score"),
    ("move_creates_relative_pin", "relative_pin_score"),
    ("move_creates_fork", "fork_score"),
    ("move_creates_skewer", "skewer_score"),
    ("move_creates_capture", "mvv_lva_score"),
)


def _square_coordinates(move: chess.Move) -> tuple:
    """Prolog coordinates ``(from_row, from_col, to_row, to_col)`` of a move (1-based)."""
    return (
        chess.square_rank(move.from_square) + 1,
        chess.square_file(move.from_square) + 1,
        chess.square_rank(move.to_square) + 1,
        chess.square_file(move.to_square) + 1,
    )


class PrologPatternDetector:
    """Detects tactical patterns of a position with the Prolog knowledge base.

    :param prolog_file: Knowledge base to consult.
    :raises ImportError: If ``pyswip`` is not installed.
    :raises RuntimeError: If SWI-Prolog is not installed.
    """

    def __init__(self, prolog_file: Union[str, Path] = KNOWLEDGE_BASE):
        try:
            from pyswip import Prolog
            from pyswip.core import SwiPrologNotFoundError
        except ImportError as error:
            raise ImportError(
                "PrologPatternDetector needs pyswip: pip install -e '.[prolog]'"
            ) from error
        try:
            self.prolog = Prolog()
        except SwiPrologNotFoundError as error:
            raise RuntimeError(
                "PrologPatternDetector needs SWI-Prolog (apt install swi-prolog)"
            ) from error
        self.prolog.consult(str(prolog_file))

    def board_to_facts(self, board: chess.Board) -> List[str]:
        """Convert a position into Prolog facts.

        :param board: Position to describe.
        :return: Facts such as ``piece(n, white, 1, 2)``, ``to_move(white)`` or ``can_castle(...)``.
        """
        facts = []
        for square, piece in board.piece_map().items():
            color = "white" if piece.color else "black"
            row = chess.square_rank(square) + 1
            col = chess.square_file(square) + 1
            facts.append(f"piece({piece.symbol().lower()}, {color}, {row}, {col})")

        facts.append(f"to_move({'white' if board.turn else 'black'})")

        for color, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
            if board.has_kingside_castling_rights(color):
                facts.append(f"can_castle({name}, kingside)")
            if board.has_queenside_castling_rights(color):
                facts.append(f"can_castle({name}, queenside)")

        if board.ep_square:
            facts.append(
                f"en_passant({chess.square_rank(board.ep_square) + 1}, "
                f"{chess.square_file(board.ep_square) + 1})"
            )
        return facts

    def clear_dynamic_facts(self) -> None:
        """Remove the facts of the previously loaded position."""
        for predicate, arity in (
            ("piece", 4),
            ("to_move", 1),
            ("can_castle", 2),
            ("en_passant", 2),
            ("move", 4),
        ):
            arguments = ",".join("_" * arity)
            list(self.prolog.query(f"retractall({predicate}({arguments}))"))

    def load_position(self, board: chess.Board) -> None:
        """Load a position and its legal moves into the knowledge base.

        :param board: Position to load.
        """
        self.clear_dynamic_facts()
        for fact in self.board_to_facts(board):
            self.prolog.assertz(fact)
        for move in board.legal_moves:
            self.prolog.assertz("move({}, {}, {}, {})".format(*_square_coordinates(move)))

    def detect_move_patterns(
        self, board: chess.Board, moves: List[chess.Move]
    ) -> Dict[chess.Move, Dict[str, Any]]:
        """Find which of ``moves`` create tactical patterns.

        The position is loaded once and every pattern is queried once for all moves.

        :param board: Current position.
        :param moves: Legal moves to classify.
        :return: For every move a dictionary with ``absolute_pin_score``, ``relative_pin_score``,
            ``fork_score``, ``skewer_score``, ``mvv_lva_score`` (0 when absent) and
            ``creates_promotion`` (bool).
        """
        self.load_position(board)
        by_coordinates = {_square_coordinates(move): move for move in moves}
        result = {
            move: {
                "absolute_pin_score": 0,
                "relative_pin_score": 0,
                "fork_score": 0,
                "skewer_score": 0,
                "mvv_lva_score": 0,
                "creates_promotion": False,
            }
            for move in moves
        }

        for predicate, key in _SCORED_PATTERNS:
            self._collect(
                f"{predicate}(FromR, FromC, ToR, ToC, Score)",
                by_coordinates,
                lambda move, binding, key=key: result[move].__setitem__(key, binding["Score"]),
            )
        self._collect(
            "move_creates_promotion(FromR, FromC, ToR, ToC)",
            by_coordinates,
            lambda move, binding: result[move].__setitem__("creates_promotion", True),
        )
        return result

    def _collect(
        self,
        query: str,
        by_coordinates: Dict[tuple, chess.Move],
        apply: Callable[[chess.Move, Dict[str, Any]], None],
    ) -> None:
        """Run a Prolog query and call ``apply`` for every solution that matches a move.

        :param query: Prolog goal binding ``FromR, FromC, ToR, ToC``.
        :param by_coordinates: Candidate moves indexed by their Prolog coordinates.
        :param apply: Callback ``(move, bindings)`` invoked for each match.
        """
        try:
            solutions = list(self.prolog.query(query))
        except Exception as error:  # a failing query must not abort the game
            logger.warning("Prolog query %r failed: %s", query, error)
            return
        for binding in solutions:
            move = by_coordinates.get(
                (binding["FromR"], binding["FromC"], binding["ToR"], binding["ToC"])
            )
            if move is not None:
                apply(move, binding)
