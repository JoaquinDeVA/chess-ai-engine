"""The bot: clock-aware iterative deepening run by the Rust core, with an opening book and tablebases."""

import logging
from typing import Optional, Sequence, Union

import chess

from chess_engine import _core
from chess_engine.bots.search_statistics import SearchStatistics
from chess_engine.bots.time_manager import TimeManager
from chess_engine.knowledge.endgame_tablebase import EndgameTablebase
from chess_engine.knowledge.opening_book import OpeningBook
from chess_engine.models import h1_evaluator, load_evaluator

#: Techniques of selective search the bots use by default: the ones that won in the games against
#: the reference search (docs/algorithms/selective-techniques.md). ``lmr`` is left out.
DEFAULT_TECHNIQUES = ("incremental", "tt_move", "history", "pvs", "aspiration", "null_move")

#: Defaults of the search, defined in the Rust core.
DEFAULT_ESTIMATOR = _core.DEFAULT_ESTIMATOR
DEFAULT_MIN_DEPTH = _core.DEFAULT_MIN_DEPTH
MAX_DEPTH = _core.MAX_DEPTH
DEFAULT_MAX_TT_ENTRIES = _core.DEFAULT_MAX_TT_ENTRIES

logger = logging.getLogger(__name__)

#: Evaluation that is the hand-written heuristic instead of a saved model.
HEURISTIC = "h1"

#: Model used when none is given: the best player in the tournaments (see docs/RESULTS.md), shipped with the repository.
DEFAULT_MODEL = "linear_pst_terms"

#: Value of ``opening_book`` and ``tablebase`` that means "the ones shipped with the project".
SHIPPED = "shipped"


def fallback_move(board: chess.Board) -> chess.Move:
    """Cheap legal move used when a search returns nothing (best capture, else first move).

    The best capture is the one with the most valuable victim and, among those, the least
    valuable attacker.

    :param board: Current position.
    :return: A legal move, or the null move if there is none.
    """
    legal = list(board.legal_moves)
    if not legal:
        return chess.Move.null()

    def score(move: chess.Move) -> int:
        if not board.is_capture(move):
            return 0
        victim = (
            chess.PAWN if board.is_en_passant(move) else board.piece_at(move.to_square).piece_type
        )
        return 10 * victim - board.piece_at(move.from_square).piece_type

    return max(legal, key=score)  # the first of the best: quiet moves keep their order


#: Quiescence plies used by default by the search bots (see docs/ALGORITHMS.md).
DEFAULT_QUIESCENCE_PLIES = _core.DEFAULT_QUIESCENCE_PLIES


class SearchBot:
    """Clock-aware iterative deepening alpha-beta, run by the Rust core.

    Every move gets a time budget from the :class:`TimeManager`; the search deepens until the
    budget is used up or ``max_depth`` is reached; without ``max_depth`` only the clock (and
    a technical cap of the core) stops it. A transposition table (bounded) and the
    killer moves persist during a game and are cleared when a new game starts: their values are
    from the point of view of the colour the bot played.

    :param evaluation: ``"h1"`` (the hand-written heuristic), a model name or ``name@tag`` for a
        tagged artifact.
    :param name: Bot name shown in games and tournaments (default: the evaluation).
    :param max_depth: Deepest iteration to run; ``None`` (default) lets the time decide. A fixed
        depth is for benchmarks and tests.
    :param time_manager: Time budgeting policy.
    :param max_tt_entries: Size cap of the transposition table.
    :param quiescence_plies: Plies of capture/promotion search at the horizon (0 = off; default 4).
    :param min_depth: Iterations always completed whatever the clock estimate says (default 1).
    :param estimator: Rule that predicts the duration of the next iteration (see
        ``rust/src/search/estimators.rs``).
    :param tablebase: Endgame tablebases; positions with up to five pieces are played from them
        without searching. ``SHIPPED`` (default) uses ``data/knowledge/endgame`` if it exists;
        ``None`` switches them off, so the bot searches every position.
    :param opening_book: Opening book; while the game follows it the bot plays its moves, and once
        the game leaves it the book is not consulted again. ``SHIPPED`` (default) uses
        ``data/opening_book.json``; ``None`` switches it off.
    :param threads: Threads that search (default 1). With more, they share a lock-free table
        (Lazy SMP), the search is no longer deterministic and the evaluation must run in the core.
    :param hash_mb: Size in MB of the table shared by several threads (ignored with one).
    :param techniques: Techniques to switch on, from ``_core.TECHNIQUES``: ``incremental``,
        ``tt_move``, ``history``, ``pvs``, ``aspiration``, ``null_move`` (the default) and ``lmr``.
        An empty tuple is the reference search.
    :ivar disabled: Techniques of the search to switch off when measuring what each one saves:
        any of ``"table"``, ``"killers"`` and ``"ordering"`` (none by default).
    """

    def __init__(
        self,
        evaluation: str = DEFAULT_MODEL,
        name: Optional[str] = None,
        max_depth: Optional[int] = None,
        time_manager: Optional[TimeManager] = None,
        max_tt_entries: int = DEFAULT_MAX_TT_ENTRIES,
        quiescence_plies: int = DEFAULT_QUIESCENCE_PLIES,
        min_depth: int = DEFAULT_MIN_DEPTH,
        estimator: str = DEFAULT_ESTIMATOR,
        tablebase: Union[EndgameTablebase, None, str] = SHIPPED,
        opening_book: Union[OpeningBook, None, str] = SHIPPED,
        threads: int = 1,
        hash_mb: int = 64,
        techniques: Sequence[str] = DEFAULT_TECHNIQUES,
    ):
        self.evaluation = evaluation
        self.name = name or evaluation
        self.max_depth = MAX_DEPTH if max_depth is None else max_depth
        self.time_manager = time_manager or TimeManager()
        self.max_tt_entries = max_tt_entries
        self.quiescence_plies = quiescence_plies
        self.min_depth = min_depth
        self.estimator = estimator
        self.tablebase = EndgameTablebase.shipped() if tablebase == SHIPPED else tablebase
        self.opening_book = OpeningBook.shipped() if opening_book == SHIPPED else opening_book
        self.in_book = self.opening_book is not None
        self.disabled: tuple = ()
        self.techniques = tuple(techniques)
        self.threads = threads
        self.searcher = _core.Searcher(max_tt_entries, threads, hash_mb)
        self._evaluator = h1_evaluator() if evaluation == HEURISTIC else load_evaluator(evaluation)
        self.turn_budget = self.time_manager.turn_budget(float("inf"))
        self.last_search_stats: Optional[SearchStatistics] = None

    def on_game_start(self, color: chess.Color, opponent_name: str) -> None:
        """Clear everything learned in the previous game.

        :param color: Colour this bot plays.
        :param opponent_name: Name of the opponent.
        """
        self.searcher.clear()
        self.last_search_stats = None
        self.in_book = self.opening_book is not None

    def search(self, board: chess.Board, time_limit: float) -> Optional[chess.Move]:
        """Run the iterative-deepening search for the position.

        :param board: Current position.
        :param time_limit: Seconds left on the clock.
        :return: The best move found, or None if the search could not produce one.
        """
        self.turn_budget = self.time_manager.turn_budget(time_limit)
        uci, raw = self.searcher.search(
            board.root().fen(),
            [move.uci() for move in board.move_stack],
            self._evaluator,
            self.max_depth,
            self.turn_budget,
            self.time_manager.hard_limit(self.turn_budget),
            self.quiescence_plies,
            self.min_depth,
            self.estimator,
            list(self.disabled),
            list(self.techniques),
        )
        stats = SearchStatistics.from_core(raw)
        self.last_search_stats = stats
        logger.debug(
            "%s searched %d nodes to ply %d",
            self.name,
            stats.nodes_visited,
            stats.max_depth_reached,
        )
        return chess.Move.from_uci(uci) if uci else None

    def get_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """Return the best move found within the time budget.

        :param board: Current position.
        :param time_limit: Seconds left on the clock.
        :return: A legal move.
        """
        if self.in_book:
            book_move = self.opening_book.lookup(board)
            if book_move is not None:
                self.last_search_stats = SearchStatistics(0, 0, 0, False, 0, (), "book", 0.0)
                return book_move
            self.in_book = False
        if self.tablebase is not None:
            move = self.tablebase.best_move(board)
            if move is not None:
                self.last_search_stats = SearchStatistics(0, 0, 0, False, 0, (), "tablebase", 0.0)
                return move
        move = self.search(board, time_limit)
        return move if move is not None and move in board.legal_moves else fallback_move(board)
