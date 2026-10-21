"""Opening book, then iterative-deepening alpha-beta evaluated by a learned model."""

from typing import Any, Optional, Union

import chess

from chess_engine.bots._search_bots import AUTHOR, IterativeDeepeningBot
from chess_engine.knowledge.opening_book import DEFAULT_BOOK_PATH, OpeningBook
from chess_engine.models import EvaluationModel, load_model
from chess_engine.models.evaluator import ModelEvaluator
from chess_engine.search.chess_game_state import ChessGameState
from chess_engine.search.time_management import TimeManager

#: Model used when none is given: a course model whose weights are loaded (not retrained; see docs/RESULTS.md).
DEFAULT_MODEL = "mlp_features"


class IterativeModelBookBot(IterativeDeepeningBot):
    """Plays book moves while the game follows the opening book, then searches.

    Once the game leaves the book it never consults it again. The search is the clock-aware
    iterative deepening of :class:`IterativeDeepeningBot` with the learned evaluation.

    :param model: Registered model name, or an already loaded :class:`EvaluationModel`.
    :param opening_book: Book to use; the one shipped in ``data/`` by default (an empty book
        if that file is missing).
    :param max_depth: Deepest iteration to run.
    :param name: Bot name shown in games and tournaments.
    :param time_manager: Time budgeting policy.
    :param search_options: Other options of :class:`IterativeDeepeningBot` (``quiescence_plies``,
        ``min_depth``, ``estimator``, ``max_tt_entries``, ``tablebase``).
    :raises FileNotFoundError: If the saved model does not exist yet.
    """

    def __init__(
        self,
        model: Union[str, EvaluationModel] = DEFAULT_MODEL,
        opening_book: Optional[OpeningBook] = None,
        max_depth: int = 3,
        name: str = "IterativeModelBookBot",
        time_manager: Optional[TimeManager] = None,
        **search_options: Any,
    ):
        super().__init__(name, AUTHOR, max_depth, time_manager, **search_options)
        self.ml_model = load_model(model) if isinstance(model, str) else model
        self.evaluator = ModelEvaluator(self.ml_model)
        if opening_book is None:
            opening_book = OpeningBook.load() if DEFAULT_BOOK_PATH.exists() else OpeningBook({})
        self.opening_book = opening_book
        self.in_book = True

    def evaluate(self, state: ChessGameState, player: chess.Color) -> float:
        """Expected score of ``player`` according to the model.

        :param state: Position to evaluate.
        :param player: Player the value is for.
        :return: Value in ``[0.01, 0.99]``.
        """
        return self.evaluator(state, player)

    def on_game_start(self, color: chess.Color, opponent_name: str) -> None:
        """Start a new game: clear the search memory and re-enable the opening book.

        :param color: Colour this bot plays.
        :param opponent_name: Name of the opponent.
        """
        super().on_game_start(color, opponent_name)
        self.in_book = True

    def get_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """Play the book move if the position is in the book, otherwise search.

        :param board: Current position.
        :param time_limit: Seconds left on the clock.
        :return: A legal move.
        """
        if self.in_book:
            book_move = self.opening_book.lookup(board)
            if book_move is not None:
                return book_move
            self.in_book = False
        return super().get_move(board, time_limit)

    def get_info(self):
        """Describe the bot.

        :return: Bot information including the model, book size and depth cap.
        """
        info = super().get_info()
        info.update(
            description=(
                f"Opening book + iterative deepening evaluated by model '{self.ml_model.name}'"
            ),
            model=self.ml_model.name,
            book_positions=len(self.opening_book),
            max_depth=self.max_depth,
        )
        return info
