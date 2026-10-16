import random
import time
from typing import Any, Callable, Dict, List

import chess
import chess.engine
from pyswip import Prolog

from adversarial_search.chess_game_state import ChessGameState
from adversarial_search.chess_problem import ChessProblem
from adversarial_search.game_algorithms import heuristic_alphabeta_search
from bots.chess_bot import ChessBot
from bots.heuristic_alphabeta_bot import h3


class PrologPatternDetector:
    """Handles Prolog-based tactical pattern detection for chess positions."""

    def __init__(self, prolog_file: str = "bots/chess_patterns.pl"):
        """Initialize Prolog engine and load knowledge base.

        :param prolog_file: Path to the Prolog knowledge base file
        :type prolog_file: str
        """
        self.prolog = Prolog()
        self.prolog.consult(prolog_file)

    def board_to_facts(self, board: chess.Board) -> List[str]:
        """Convert a chess board position to Prolog facts.

        :param board: Current chess position
        :type board: chess.Board
        :return: List of Prolog facts representing the position
        :rtype: List[str]
        """
        facts = []

        # Add piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = chess.square_rank(square) + 1  # 1-8
                col = chess.square_file(square) + 1  # 1-8 (a-h mapped to 1-8)
                color = "white" if piece.color else "black"
                piece_type = piece.symbol().lower()

                # piece(Type, Color, Row, Col)
                facts.append(f"piece({piece_type}, {color}, {row}, {col})")

        # Add current player
        current_player = "white" if board.turn else "black"
        facts.append(f"to_move({current_player})")

        # Add castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            facts.append("can_castle(white, kingside)")
        if board.has_queenside_castling_rights(chess.WHITE):
            facts.append("can_castle(white, queenside)")
        if board.has_kingside_castling_rights(chess.BLACK):
            facts.append("can_castle(black, kingside)")
        if board.has_queenside_castling_rights(chess.BLACK):
            facts.append("can_castle(black, queenside)")

        # Add en passant square if it exists
        if board.ep_square:
            ep_row = chess.square_rank(board.ep_square) + 1
            ep_col = chess.square_file(board.ep_square) + 1
            facts.append(f"en_passant({ep_row}, {ep_col})")

        return facts

    def clear_dynamic_facts(self) -> None:
        """Clear all dynamic facts from the Prolog knowledge base.

        Uses retractall to ensure all instances of dynamic predicates are removed,
        preparing for the next position to be loaded.
        """
        # Define predicates and their arities to retract
        predicates_to_clear = [
            ("piece", 4),  # piece(Type, Color, Row, Col)
            ("to_move", 1),  # to_move(Color)
            ("can_castle", 2),  # can_castle(Color, Side)
            ("en_passant", 2),  # en_passant(Row, Col)
            ("move", 4),  # move(FromRow, FromCol, ToRow, ToCol)
        ]

        for pred_name, arity in predicates_to_clear:
            try:
                # Build retractall query with appropriate number of arguments
                args = ",".join("_" * arity)
                query = f"retractall({pred_name}({args}))"
                list(self.prolog.query(query))
            except Exception as e:
                # Silently ignore errors - predicate may not exist or may already be empty
                pass

    def load_position(self, board: chess.Board) -> None:
        """Load a chess position into Prolog knowledge base.

        :param board: Chess position to load
        :type board: chess.Board
        """
        # Clear previous position
        self.clear_dynamic_facts()

        # Assert new facts
        facts = self.board_to_facts(board)
        for fact in facts:
            self.prolog.assertz(fact)

        # Assert all legal moves as facts
        for move in board.legal_moves:
            from_row = chess.square_rank(move.from_square) + 1
            from_col = chess.square_file(move.from_square) + 1
            to_row = chess.square_rank(move.to_square) + 1
            to_col = chess.square_file(move.to_square) + 1
            move_fact = f"move({from_row}, {from_col}, {to_row}, {to_col})"
            self.prolog.assertz(move_fact)

    def detect_move_patterns(
        self, board: chess.Board, moves: List[chess.Move]
    ) -> Dict[chess.Move, Dict[str, Any]]:
        """Efficiently detect which moves create tactical patterns.

        This method loads the position once and queries pattern creation for all moves,
        avoiding the expensive board copying and pattern detection for each move.

        :param board: Current chess position
        :type board: chess.Board
        :param moves: List of legal moves to evaluate
        :type moves: List[chess.Move]
        :return: Dictionary mapping moves to their tactical properties:
            {move: {
                "absolute_pin_score": int (0 if no absolute pin),
                "relative_pin_score": int (0 if no relative pin),
                "fork_score": int (0 if no fork),
                "skewer_score": int (0 if no skewer),
                "mvv_lva_score": int (0 if not a capture),
                "creates_promotion": bool
            }}
        :rtype: Dict[chess.Move, Dict[str, Any]]
        """
        # Load the position into Prolog
        self.load_position(board)

        # Build a mapping from (from_row, from_col, to_row, to_col) to move
        move_map = {}
        for move in moves:
            from_row = chess.square_rank(move.from_square) + 1
            from_col = chess.square_file(move.from_square) + 1
            to_row = chess.square_rank(move.to_square) + 1
            to_col = chess.square_file(move.to_square) + 1
            move_map[(from_row, from_col, to_row, to_col)] = move

        # Initialize result dictionary
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

        # Query for moves that create absolute pins
        try:
            absolute_pin_moves = list(
                self.prolog.query(
                    "move_creates_absolute_pin(FromR, FromC, ToR, ToC, Score)"
                )
            )
            for binding in absolute_pin_moves:
                coords = (
                    binding["FromR"],
                    binding["FromC"],
                    binding["ToR"],
                    binding["ToC"],
                )
                if coords in move_map:
                    result[move_map[coords]]["absolute_pin_score"] = binding["Score"]
        except Exception as e:
            pass

        # Query for moves that create relative pins
        try:
            relative_pin_moves = list(
                self.prolog.query(
                    "move_creates_relative_pin(FromR, FromC, ToR, ToC, Score)"
                )
            )
            for binding in relative_pin_moves:
                coords = (
                    binding["FromR"],
                    binding["FromC"],
                    binding["ToR"],
                    binding["ToC"],
                )
                if coords in move_map:
                    result[move_map[coords]]["relative_pin_score"] = binding["Score"]
        except Exception as e:
            pass

        # Query for moves that create forks
        try:
            fork_moves = list(
                self.prolog.query("move_creates_fork(FromR, FromC, ToR, ToC, Score)")
            )
            for binding in fork_moves:
                coords = (
                    binding["FromR"],
                    binding["FromC"],
                    binding["ToR"],
                    binding["ToC"],
                )
                if coords in move_map:
                    result[move_map[coords]]["fork_score"] = binding["Score"]
        except Exception as e:
            pass

        # Query for moves that create skewers
        try:
            skewer_moves = list(
                self.prolog.query("move_creates_skewer(FromR, FromC, ToR, ToC, Score)")
            )
            for binding in skewer_moves:
                coords = (
                    binding["FromR"],
                    binding["FromC"],
                    binding["ToR"],
                    binding["ToC"],
                )
                if coords in move_map:
                    result[move_map[coords]]["skewer_score"] = binding["Score"]
        except Exception as e:
            pass

        # Query for moves that are captures (with MVV-LVA score)
        try:
            capture_moves = list(
                self.prolog.query("move_creates_capture(FromR, FromC, ToR, ToC, Score)")
            )
            for binding in capture_moves:
                coords = (
                    binding["FromR"],
                    binding["FromC"],
                    binding["ToR"],
                    binding["ToC"],
                )
                if coords in move_map:
                    result[move_map[coords]]["mvv_lva_score"] = binding["Score"]
        except Exception as e:
            pass

        # Query for moves that are promotions
        try:
            promotion_moves = list(
                self.prolog.query("move_creates_promotion(FromR, FromC, ToR, ToC)")
            )
            for binding in promotion_moves:
                coords = (
                    binding["FromR"],
                    binding["FromC"],
                    binding["ToR"],
                    binding["ToC"],
                )
                if coords in move_map:
                    result[move_map[coords]]["creates_promotion"] = True
        except Exception as e:
            pass

        return result


class TacticalAlphaBetaBot(ChessBot):
    """
    Chess bot using heuristic alpha-beta search with configurable parameters.
    """

    def __init__(
        self,
        eval_function: Callable[[ChessGameState, chess.Color], float] = h3,
        max_depth: int = 3,
        time_buffer: float = 0.1,
        name = "TacticalAlphaBetaBot",
    ):
        """
        Initialize the heuristic alpha-beta bot.

        :param eval_function: Evaluation function (state, player) -> float
        :type eval_function: Callable[[ChessGameState, chess.Color], float]
        :param max_depth: Maximum search depth
        :type max_depth: int
        :param time_buffer: Time buffer to avoid timeout (seconds)
        :type time_buffer: float
        """
        super().__init__(name , "Joaquin de Vicente Abad")
        self.eval_function = eval_function
        self.max_depth = max_depth
        self.time_buffer = time_buffer
        self.pattern_detector = PrologPatternDetector()

        # Statistics tracking
        self.last_search_stats = None  # Either None or SearchStatistics
        self.total_nodes_visited = 0
        self.total_pruning_count = 0
        self.move_count = 0

    def on_game_start(self, color: chess.Color, opponent_name: str) -> None:
        """
        Called when a new game starts. Resets statistics.

        :param color: Our color this game
        :type color: chess.Color
        :param opponent_name: Name of opponent
        :type opponent_name: str
        """
        # Reset statistics
        self.last_search_stats = None
        self.total_nodes_visited = 0
        self.total_pruning_count = 0
        self.move_count = 0

    def on_game_end(self, result: str, board: chess.Board) -> None:
        """
        Called when a game ends. Prints search statistics.

        :param result: Game result (1-0, 0-1, 1/2-1/2)
        :type result: str
        :param board: Final board position
        :type board: chess.Board
        """
        print(f"\n{self.name} Search Statistics:")
        print("=" * 50)
        stats = self.get_search_statistics()
        print(f"Total moves made: {stats['move_count']}")
        print(f"Total nodes visited: {stats['total_nodes_visited']}")
        print(f"Total pruning events: {stats['total_pruning_count']}")
        print(f"Average nodes per move: {stats['avg_nodes_per_move']:.1f}")
        print(f"Average pruning per move: {stats['avg_pruning_per_move']:.1f}")
        if stats["move_count"] > 0:
            pruning_rate = (
                (stats["total_pruning_count"] / stats["total_nodes_visited"] * 100)
                if stats["total_nodes_visited"] > 0
                else 0
            )
            print(f"Pruning rate: {pruning_rate:.2f}%")
        print("=" * 50)

    def order_moves(
        self, board: chess.Board, moves: List[chess.Move]
    ) -> List[chess.Move]:
        """Order moves for better alpha-beta pruning.

        Uses tactical pattern detection with dynamic scoring:
        - Absolute pins: 50 + (score * 3), range: 56-209
        - Relative pins: 30 + (score * 3), range: 36-51
        - Forks: 100 + (score * 3), range: 121-418
        - Skewers: 80 + (score * 3), range: 89-380
        - Captures: score * 3, range: 3-267 (MVV-LVA)
        - Promotions: +800
        - Checks: +50 (includes discovered checks)

        Scores are dynamically calculated in Prolog based on piece values,
        then scaled and combined with baselines to create a tactical hierarchy:
        King-involved tactics > Good captures > Minor tactics ≈ Medium captures

        This method efficiently detects patterns by loading the position once
        and querying which moves create tactical patterns, avoiding expensive
        board copying for each move.

        :param board: Current position
        :type board: chess.Board
        :param moves: List of legal moves
        :type moves: List[chess.Move]
        :return: Ordered list of moves (best moves first)
        :rtype: List[chess.Move]
        """
        # Efficiently detect which moves create tactical patterns
        move_patterns = self.pattern_detector.detect_move_patterns(board, moves)

        move_scores = []

        for move in moves:
            score = 0.0

            patterns = move_patterns.get(move, {})

            # Tactical pattern scores (from Prolog - baseline + 3x dynamic scoring)
            absolute_pin_score = patterns.get("absolute_pin_score", 0)
            if absolute_pin_score > 0:
                score += 50 + absolute_pin_score * 3

            relative_pin_score = patterns.get("relative_pin_score", 0)
            if relative_pin_score > 0:
                score += 30 + relative_pin_score * 3

            fork_score = patterns.get("fork_score", 0)
            if fork_score > 0:
                score += 100 + fork_score * 3

            skewer_score = patterns.get("skewer_score", 0)
            if skewer_score > 0:
                score += 80 + skewer_score * 3

            # MVV-LVA capture scoring (from Prolog - 3x scaling, no baseline)
            mvv_lva_score = patterns.get("mvv_lva_score", 0)
            if mvv_lva_score > 0:
                score += mvv_lva_score * 3

            # Promotion bonus (from Prolog)
            if patterns.get("creates_promotion", False):
                score += 800

            # Check bonus (from Python - includes discovered checks)
            if board.gives_check(move):
                score += 50

            move_scores.append((score, move))

        # Sort moves by score (highest first)
        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in move_scores]

    def get_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """Get the best move using heuristic alpha-beta search.

        :param board: Current chess position
        :type board: chess.Board
        :param time_limit: Maximum time to think in seconds
        :type time_limit: float
        :return: Best move found
        :rtype: chess.Move
        """
        # Convert to our game representation
        chess_state = ChessGameState(board)
        chess_problem = ChessProblem(board)

        # Define cutoff test that checks search limits
        def cutoff_test(state: ChessGameState, depth: int, elapsed_time: float) -> bool:
            return depth >= self.max_depth

        try:
            # Use heuristic alpha-beta search
            move, stats = heuristic_alphabeta_search(
                chess_problem,
                chess_state,
                self.eval_function,
                cutoff_test,
                self.order_moves,
            )

            # Track statistics
            self.last_search_stats = stats
            self.total_nodes_visited += stats.nodes_visited
            self.total_pruning_count += stats.pruning_count
            self.move_count += 1

            if move and move in board.legal_moves:
                return move

        except Exception as e:
            raise e

        # Fallback to random move if search fails
        legal_moves = list(board.legal_moves)
        if legal_moves:
            import random

            return random.choice(legal_moves)

        return chess.Move.null()

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics for analysis.

        :return: Dictionary containing search statistics
        :rtype: Dict[str, Any]
        """
        avg_nodes = (
            self.total_nodes_visited / self.move_count if self.move_count > 0 else 0
        )
        avg_pruning = (
            self.total_pruning_count / self.move_count if self.move_count > 0 else 0
        )

        stats = {
            "total_nodes_visited": self.total_nodes_visited,
            "total_pruning_count": self.total_pruning_count,
            "move_count": self.move_count,
            "avg_nodes_per_move": avg_nodes,
            "avg_pruning_per_move": avg_pruning,
        }

        if self.last_search_stats:
            stats["last_search"] = {
                "nodes_visited": self.last_search_stats.nodes_visited,
                "pruning_count": self.last_search_stats.pruning_count,
                "max_depth_reached": self.last_search_stats.max_depth_reached,
            }

        return stats

    def get_info(self) -> Dict[str, Any]:
        """Get information about this bot.

        :return: Bot information dictionary
        :rtype: Dict[str, Any]
        """
        info = super().get_info()
        info.update(
            {
                "description": "Tactical Bot",
            }
        )
        return info
