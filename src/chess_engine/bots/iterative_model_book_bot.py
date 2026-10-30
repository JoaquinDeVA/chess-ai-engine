"""Opening book, then iterative-deepening alpha-beta evaluated by a learned model."""

from typing import Any, Optional, Union

import chess

from chess_engine.bots._search_bots import AUTHOR, SHIPPED, IterativeDeepeningBot
from chess_engine.evaluation.chess_game_state import ChessGameState
from chess_engine.knowledge.opening_book import OpeningBook
from chess_engine.models import EvaluationModel, load_model
from chess_engine.models.evaluator import ModelEvaluator
from chess_engine.models.native import model_evaluator
from chess_engine.search.time_management import TimeManager

#: Model used when none is given: the best player in the tournaments (see docs/RESULTS.md), shipped with the repository.
DEFAULT_MODEL = "linear_pst_terms"


class IterativeModelBookBot(IterativeDeepeningBot):
    """Plays book moves while the game follows the opening book, then searches.

    Once the game leaves the book it never consults it again. The search is the clock-aware
    iterative deepening of :class:`IterativeDeepeningBot` with the learned evaluation.

    :param model: Registered model name, or an already loaded :class:`EvaluationModel`.
    :param opening_book: Book to use; the one shipped in ``data/`` by default (an empty book
        if that file is missing); ``None`` switches it off.
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
        opening_book: Union[OpeningBook, None, str] = SHIPPED,
        max_depth: int = 3,
        name: str = "IterativeModelBookBot",
        time_manager: Optional[TimeManager] = None,
        **search_options: Any,
    ):
        super().__init__(
            name, AUTHOR, max_depth, time_manager, opening_book=opening_book, **search_options
        )
        self.ml_model = load_model(model) if isinstance(model, str) else model
        self.evaluator = ModelEvaluator(self.ml_model)

    def native_evaluator(self):
        """The evaluation the core runs: the model itself, natively when it can be.

        :return: Evaluator for the model.
        """
        return model_evaluator(self.ml_model)

    def evaluate(self, state: ChessGameState, player: chess.Color) -> float:
        """Expected score of ``player`` according to the model.

        :param state: Position to evaluate.
        :param player: Player the value is for.
        :return: Value in ``[0.01, 0.99]``.
        """
        return self.evaluator(state, player)

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
            book_positions=len(self.opening_book) if self.opening_book is not None else 0,
            max_depth=self.max_depth,
        )
        return info
