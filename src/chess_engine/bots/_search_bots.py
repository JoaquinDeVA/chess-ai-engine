"""Base classes shared by the search-based bots.

The leading underscore keeps this file out of the bot registry (it contains no playable bot).

* :class:`IterativeDeepeningBot` - clock-aware iterative deepening with a transposition table,
  killer moves and capture ordering; subclasses only provide an evaluation.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

import chess

from chess_engine.bots.chess_bot import ChessBot
from chess_engine.search.chess_game_state import ChessGameState
from chess_engine.search.chess_problem import ChessProblem, chess_noisy_moves
from chess_engine.search.game_algorithms import (
    DEFAULT_ESTIMATOR,
    DEFAULT_MIN_DEPTH,
    SearchStatistics,
    heuristic_alphabeta_iterative_deepening_search,
)
from chess_engine.search.game_problems import GameState
from chess_engine.search.search_utils import DEFAULT_MAX_TT_ENTRIES, order_by_mvv_lva
from chess_engine.search.time_management import TimeManager

logger = logging.getLogger(__name__)

AUTHOR = "Joaquin De Vicente Abad"


def fallback_move(board: chess.Board) -> chess.Move:
    """Cheap legal move used when a search returns nothing (best capture, else first move).

    :param board: Current position.
    :return: A legal move, or the null move if there is none.
    """
    legal = list(board.legal_moves)
    if not legal:
        return chess.Move.null()
    return order_by_mvv_lva(board, legal)[0]


#: Quiescence plies used by default by the search bots (see docs/SEARCH_ALGORITHMS.md).
DEFAULT_QUIESCENCE_PLIES = 4


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
    """Clock-aware iterative deepening alpha-beta.

    Every move gets a time budget from the :class:`TimeManager`; the search deepens until the
    budget is used up or ``max_depth`` is reached. A transposition table (bounded) and the
    killer moves persist during a game and are cleared when a new game starts: their values are
    from the point of view of the colour the bot played.

    Subclasses implement :meth:`evaluate`.

    :param name: Bot name shown in games and tournaments.
    :param author: Author shown in bot listings.
    :param max_depth: Deepest iteration to run.
    :param time_manager: Time budgeting policy.
    :param max_tt_entries: Size cap of the transposition table.
    :param problem: Chess problem to search; a fresh one is created by default.
    :param quiescence_plies: Plies of capture/promotion search at the horizon (0 = off; default 4,
        measured at fixed depth: +0.06 agreement at depth 3 on 2,000 positions).
    :param min_depth: Iterations always completed whatever the clock estimate says (default 3).
    :param estimator: Rule that predicts the duration of the next iteration (see
        ``search/time_estimators.py``).
    """

    def __init__(
        self,
        name: str,
        author: str = AUTHOR,
        max_depth: int = 3,
        time_manager: Optional[TimeManager] = None,
        max_tt_entries: int = DEFAULT_MAX_TT_ENTRIES,
        problem: Optional[ChessProblem] = None,
        quiescence_plies: int = DEFAULT_QUIESCENCE_PLIES,
        min_depth: int = DEFAULT_MIN_DEPTH,
        estimator: str = DEFAULT_ESTIMATOR,
    ):
        super().__init__(name, author)
        self.max_depth = max_depth
        self.time_manager = time_manager or TimeManager()
        self.max_tt_entries = max_tt_entries
        self.quiescence_plies = quiescence_plies
        self.min_depth = min_depth
        self.estimator = estimator
        self.problem = problem if problem is not None else ChessProblem()
        self.transposition_table: Dict[Any, Any] = {}
        self.killer_moves: Dict[int, List[chess.Move]] = {}
        self.turn_budget = self.time_manager.turn_budget(float("inf"))
        self.reset_statistics()

    @abstractmethod
    def evaluate(self, state: ChessGameState, player: chess.Color) -> float:
        """Heuristic value of a non-terminal position.

        :param state: Position to evaluate.
        :param player: Player the value is for.
        :return: Expected score for ``player`` in ``[0, 1]``.
        """

    def order_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """Captures first, best MVV-LVA first.

        :param board: Position the moves are played from.
        :param moves: Legal moves.
        :return: Reordered moves.
        """
        return order_by_mvv_lva(board, moves)

    def cutoff_test(self, state: GameState, depth_left: int, time_used: float) -> bool:
        """Stop when the depth is exhausted or the move's hard time limit has passed.

        :param state: Node being visited.
        :param depth_left: Plies still to search below this node.
        :param time_used: Seconds spent on this move so far.
        :return: True when the node must be evaluated instead of expanded.
        """
        return depth_left == 0 or self.time_manager.expired(self.turn_budget, time_used)

    def on_game_start(self, color: chess.Color, opponent_name: str) -> None:
        """Clear everything learned in the previous game.

        :param color: Colour this bot plays.
        :param opponent_name: Name of the opponent.
        """
        self.transposition_table.clear()
        self.killer_moves.clear()
        self.reset_statistics()

    def search(self, board: chess.Board, time_limit: float) -> Optional[chess.Move]:
        """Run the iterative-deepening search for the position.

        :param board: Current position.
        :param time_limit: Seconds left on the clock.
        :return: The best move found, or None if the search could not produce one.
        """
        self.turn_budget = self.time_manager.turn_budget(time_limit)
        move, stats = heuristic_alphabeta_iterative_deepening_search(
            game=self.problem,
            state=ChessGameState(board),
            eval_fn=self.evaluate,
            cutoff_test=self.cutoff_test,
            killer_moves=self.killer_moves,
            transposition_table=self.transposition_table,
            order_moves_callback=self.order_moves,
            max_depth=self.max_depth,
            time_limit=self.turn_budget,
            noisy_moves=chess_noisy_moves if self.quiescence_plies else None,
            quiescence_plies=self.quiescence_plies,
            max_tt_entries=self.max_tt_entries,
            min_depth=self.min_depth,
            estimator=self.estimator,
        )
        self.record_statistics(stats)
        logger.debug(
            "%s searched %d nodes to ply %d",
            self.name,
            stats.nodes_visited,
            stats.max_depth_reached,
        )
        return move

    def get_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """Return the best move found within the time budget.

        :param board: Current position.
        :param time_limit: Seconds left on the clock.
        :return: A legal move.
        """
        move = self.search(board, time_limit)
        return move if move is not None and move in board.legal_moves else fallback_move(board)
