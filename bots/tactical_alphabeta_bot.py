"""Tactical alpha-beta bot: move ordering driven by tactical patterns found in Prolog.

A Prolog knowledge base (``knowledge/chess_patterns.pl``) recognizes pins, forks, skewers,
captures and promotions. The bot asks it which legal moves *create* such a pattern and searches
those first, which makes alpha-beta prune much more.

This module needs SWI-Prolog and the ``pyswip`` package (``pip install -e '.[prolog]'``); both
are optional for the rest of the engine.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import chess

from adversarial_search.chess_game_state import ChessGameState
from bots._search_bots import AUTHOR, FixedDepthBot
from evaluation.heuristics import h3

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE = Path(__file__).resolve().parent.parent / "knowledge" / "chess_patterns.pl"

# (Prolog predicate, key in the result dictionary); all of them bind FromR, FromC, ToR, ToC, Score.
_SCORED_PATTERNS = (
    ("move_creates_absolute_pin", "absolute_pin_score"),
    ("move_creates_relative_pin", "relative_pin_score"),
    ("move_creates_fork", "fork_score"),
    ("move_creates_skewer", "skewer_score"),
    ("move_creates_capture", "mvv_lva_score"),
)

# Move-ordering weights: (baseline added when the pattern exists, multiplier of the Prolog score).
ABSOLUTE_PIN_WEIGHT = (50, 3)
RELATIVE_PIN_WEIGHT = (30, 3)
FORK_WEIGHT = (100, 3)
SKEWER_WEIGHT = (80, 3)
CAPTURE_WEIGHT = (0, 3)
PROMOTION_BONUS = 800
CHECK_BONUS = 50


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
                "TacticalAlphaBetaBot needs pyswip: pip install -e '.[prolog]'"
            ) from error
        try:
            self.prolog = Prolog()
        except SwiPrologNotFoundError as error:
            raise RuntimeError(
                "TacticalAlphaBetaBot needs SWI-Prolog (apt install swi-prolog)"
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


class TacticalAlphaBetaBot(FixedDepthBot):
    """Fixed-depth alpha-beta that searches tactically promising moves first.

    :param eval_function: Heuristic ``(state, player) -> float``; default ``h3``.
    :param max_depth: Plies to search.
    :param name: Bot name.
    :raises ImportError: If ``pyswip`` is not installed.
    :raises RuntimeError: If SWI-Prolog is not installed.
    """

    def __init__(
        self,
        eval_function: Callable[[ChessGameState, chess.Color], float] = h3,
        max_depth: int = 3,
        name: str = "TacticalAlphaBetaBot",
    ):
        super().__init__(name, AUTHOR, max_depth)
        self.eval_function = eval_function
        self.pattern_detector = PrologPatternDetector()

    def evaluate(self, state: ChessGameState, player: chess.Color) -> float:
        """Evaluate with the configured heuristic.

        :param state: Position to evaluate.
        :param player: Player the value is for.
        :return: Expected score for ``player`` in ``[0.01, 0.99]``.
        """
        return self.eval_function(state, player)

    def order_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """Order moves by tactical value, best first.

        Score = pin / fork / skewer bonuses (baseline + 3 x the Prolog score, so king-involved
        tactics rank highest) + MVV-LVA capture score + 800 for promotions + 50 for checks.

        :param board: Current position.
        :param moves: Legal moves.
        :return: The moves sorted by decreasing score.
        """
        patterns = self.pattern_detector.detect_move_patterns(board, moves)

        def score(move: chess.Move) -> float:
            found = patterns.get(move, {})
            total = 0.0
            for key, (baseline, factor) in (
                ("absolute_pin_score", ABSOLUTE_PIN_WEIGHT),
                ("relative_pin_score", RELATIVE_PIN_WEIGHT),
                ("fork_score", FORK_WEIGHT),
                ("skewer_score", SKEWER_WEIGHT),
                ("mvv_lva_score", CAPTURE_WEIGHT),
            ):
                value = found.get(key, 0)
                if value > 0:
                    total += baseline + value * factor
            if found.get("creates_promotion"):
                total += PROMOTION_BONUS
            if board.gives_check(move):
                total += CHECK_BONUS
            return total

        return sorted(moves, key=score, reverse=True)

    def on_game_end(self, result: str, board: chess.Board) -> None:
        """Log the search statistics of the game.

        :param result: Game result (``1-0``, ``0-1`` or ``1/2-1/2``).
        :param board: Final position.
        """
        stats = self.get_search_statistics()
        logger.info("%s search statistics: %s", self.name, stats)

    def get_info(self):
        """Describe the bot.

        :return: Bot information.
        """
        info = super().get_info()
        info.update(
            description="Alpha-beta with Prolog tactical-pattern move ordering",
            max_depth=self.max_depth,
        )
        return info
