"""
Non-Tactical Alpha-Beta Bot

A simplified version of TacticalAlphaBetaBot that does NOT use the Prolog-based
pattern detector for move ordering. It only relies on heuristic alpha-beta search.
"""

import random
import chess
from adversarial_search.chess_game_state import ChessGameState
from adversarial_search.chess_problem import ChessProblem
from adversarial_search.game_algorithms import heuristic_alphabeta_search
from bots.tactical_alphabeta_bot import TacticalAlphaBetaBot
from bots.heuristic_alphabeta_bot import h3


class NonTacticalAlphaBetaBot(TacticalAlphaBetaBot):
    """A non-tactical version of the tactical alpha-beta bot."""

    def __init__(self, eval_function=h3, max_depth: int = 3, name: str = "NonTacticalAlphaBetaBot"):

        # Skip calling TacticalAlphaBetaBot.__init__ directly to avoid creating PrologPatternDetector
        super(TacticalAlphaBetaBot, self).__init__(name, "Joaquin de Vicente Abad")
        self.eval_function = eval_function
        self.max_depth = max_depth
        self.last_search_stats = None
        self.total_nodes_visited = 0
        self.total_pruning_count = 0
        self.move_count = 0

    def get_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """Get the best move using heuristic alpha-beta search (no tactical ordering)."""
        chess_state = ChessGameState(board)
        chess_problem = ChessProblem(board)

        def cutoff_test(state: ChessGameState, depth: int, elapsed_time: float) -> bool:
            return depth >= self.max_depth

        try:
            move, stats = heuristic_alphabeta_search(
                chess_problem,
                chess_state,
                self.eval_function,
                cutoff_test
            )

            self.last_search_stats = stats
            self.total_nodes_visited += stats.nodes_visited
            self.total_pruning_count += stats.pruning_count
            self.move_count += 1

            if move and move in board.legal_moves:
                return move

        except Exception as e:
            print(f"[Non-Tactical Bot Error] {e}")

        # Fallback: choose a random move if search fails
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves) if legal_moves else chess.Move.null()
