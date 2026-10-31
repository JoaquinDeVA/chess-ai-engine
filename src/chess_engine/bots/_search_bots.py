"""Base classes shared by the search-based bots.

The leading underscore keeps this file out of the bot registry (it contains no playable bot).

* :class:`IterativeDeepeningBot` - clock-aware iterative deepening with a transposition table,
  killer moves and capture ordering; subclasses only provide an evaluation.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence, Union

import chess

from chess_engine import _core
from chess_engine.bots.chess_bot import ChessBot
from chess_engine.knowledge.endgame_tablebase import EndgameTablebase
from chess_engine.knowledge.opening_book import OpeningBook
from chess_engine.search.statistics import SearchStatistics
from chess_engine.search.time_management import TimeManager

#: Techniques of selective search the bots use by default: the ones that won in the games against
#: the reference search (docs/algorithms/selective-techniques.md). ``lmr`` is left out.
DEFAULT_TECHNIQUES = ("incremental", "tt_move", "history", "pvs", "aspiration", "null_move")

#: Defaults of the search, defined in the Rust core.
DEFAULT_ESTIMATOR = _core.DEFAULT_ESTIMATOR
DEFAULT_MIN_DEPTH = _core.DEFAULT_MIN_DEPTH
MAX_DEPTH = _core.MAX_DEPTH
DEFAULT_MAX_TT_ENTRIES = _core.DEFAULT_MAX_TT_ENTRIES

logger = logging.getLogger(__name__)

AUTHOR = "Joaquin De Vicente Abad"

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


class _StatisticsMixin:
    """Accumulates search statistics over a game."""

    last_search_stats: Optional[SearchStatistics]
    total_nodes_visited: int
    total_pruning_count: int
    move_count: int

    def reset_statistics(self) -> None:
        """Forget the statistics of previous games."""
        self.last_search_stats = None
        self.total_nodes_visited = 0
        self.total_pruning_count = 0
        self.move_count = 0

    def record_statistics(self, stats: SearchStatistics) -> None:
        """Add the statistics of one search.

        :param stats: Statistics returned by the search.
        """
        self.last_search_stats = stats
        self.total_nodes_visited += stats.nodes_visited
        self.total_pruning_count += stats.pruning_count
        self.move_count += 1

    def get_search_statistics(self) -> Dict[str, Any]:
        """Search statistics of the current game, for analysis.

        :return: Totals, per-move averages and the details of the last search.
        """
        moves = self.move_count
        stats: Dict[str, Any] = {
            "total_nodes_visited": self.total_nodes_visited,
            "total_pruning_count": self.total_pruning_count,
            "move_count": moves,
            "avg_nodes_per_move": self.total_nodes_visited / moves if moves else 0,
            "avg_pruning_per_move": self.total_pruning_count / moves if moves else 0,
        }
        if self.last_search_stats:
            stats["last_search"] = {
                "nodes_visited": self.last_search_stats.nodes_visited,
                "pruning_count": self.last_search_stats.pruning_count,
                "max_depth_reached": self.last_search_stats.max_depth_reached,
            }
        return stats


class IterativeDeepeningBot(_StatisticsMixin, ChessBot):
    """Clock-aware iterative deepening alpha-beta, run by the Rust core.

    Every move gets a time budget from the :class:`TimeManager`; the search deepens until the
    budget is used up or ``max_depth`` is reached; without ``max_depth`` only the clock (and
    a technical cap of the core) stops it. A transposition table (bounded) and the
    killer moves persist during a game and are cleared when a new game starts: their values are
    from the point of view of the colour the bot played.

    Subclasses implement :meth:`evaluate` and, to evaluate inside the core, :meth:`native_evaluator`.

    :param name: Bot name shown in games and tournaments.
    :param author: Author shown in bot listings.
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
        name: str,
        author: str = AUTHOR,
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
        super().__init__(name, author)
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
        self._native: Optional[_core.Evaluator] = None
        self.turn_budget = self.time_manager.turn_budget(float("inf"))
        self.reset_statistics()

    @abstractmethod
    def native_evaluator(self) -> "_core.Evaluator":
        """The evaluation the Rust core runs.

        :return: Evaluator.
        """

    def on_game_start(self, color: chess.Color, opponent_name: str) -> None:
        """Clear everything learned in the previous game.

        :param color: Colour this bot plays.
        :param opponent_name: Name of the opponent.
        """
        self.searcher.clear()
        self.reset_statistics()
        self.in_book = self.opening_book is not None

    def search(self, board: chess.Board, time_limit: float) -> Optional[chess.Move]:
        """Run the iterative-deepening search for the position.

        :param board: Current position.
        :param time_limit: Seconds left on the clock.
        :return: The best move found, or None if the search could not produce one.
        """
        if self._native is None:
            self._native = self.native_evaluator()
        self.turn_budget = self.time_manager.turn_budget(time_limit)
        uci, raw = self.searcher.search(
            board.root().fen(),
            [move.uci() for move in board.move_stack],
            self._native,
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
        self.record_statistics(stats)
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
