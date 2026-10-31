"""Opening book, then iterative-deepening alpha-beta evaluated by a learned model."""

from typing import Any, Optional, Union

from chess_engine.bots._search_bots import AUTHOR, SHIPPED, IterativeDeepeningBot
from chess_engine.knowledge.opening_book import OpeningBook
from chess_engine.models import load_evaluator
from chess_engine.search.time_management import TimeManager

#: Model used when none is given: the best player in the tournaments (see docs/RESULTS.md), shipped with the repository.
DEFAULT_MODEL = "linear_pst_terms"


class IterativeModelBookBot(IterativeDeepeningBot):
    """Plays book moves while the game follows the opening book, then searches.

    Once the game leaves the book it never consults it again. The search is the clock-aware
    iterative deepening of :class:`IterativeDeepeningBot` with the learned evaluation.

    :param model: Model name, or ``name@tag`` for a tagged artifact.
    :param opening_book: Book to use; the one shipped in ``data/`` by default (an empty book
        if that file is missing); ``None`` switches it off.
    :param max_depth: Deepest iteration to run; ``None`` (default) lets the time decide.
    :param name: Bot name shown in games and tournaments.
    :param time_manager: Time budgeting policy.
    :param search_options: Other options of :class:`IterativeDeepeningBot` (``quiescence_plies``,
        ``min_depth``, ``estimator``, ``max_tt_entries``, ``tablebase``).
    :raises FileNotFoundError: If the saved model does not exist yet.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        opening_book: Union[OpeningBook, None, str] = SHIPPED,
        max_depth: Optional[int] = None,
        name: str = "IterativeModelBookBot",
        time_manager: Optional[TimeManager] = None,
        **search_options: Any,
    ):
        super().__init__(
            name, AUTHOR, max_depth, time_manager, opening_book=opening_book, **search_options
        )
        self.model = model
        self._evaluator = load_evaluator(model)

    def native_evaluator(self):
        """The evaluation the core runs: the model itself.

        :return: Evaluator for the model.
        """
        return self._evaluator

    def get_info(self):
        """Describe the bot.

        :return: Bot information including the model, book size and depth cap.
        """
        info = super().get_info()
        info.update(
            description=(f"Opening book + iterative deepening evaluated by model '{self.model}'"),
            model=self.model,
            book_positions=len(self.opening_book) if self.opening_book is not None else 0,
            max_depth=self.max_depth,
        )
        return info
