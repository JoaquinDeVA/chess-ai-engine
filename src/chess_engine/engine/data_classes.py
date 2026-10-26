"""
Data classes for the chess engine.

This module contains all the data structures used throughout the chess engine
(tournament configuration, game and tournament results, PGN export).

Based on the AI course framework (Inteligencia Artificial e Ingeniería del Conocimiento,
CEU San Pablo University).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import chess
import chess.pgn


@dataclass
class TournamentConfig:
    """
    Tournament configuration settings.

    :param total_game_time: Total time per player in seconds
    :type total_game_time: float
    :param time_increment: Time increment per move in seconds
    :type time_increment: float
    :param rounds: Number of tournament rounds
    :type rounds: int
    :param scoring: Scoring system dictionary
    :type scoring: Dict[str, float]
    """

    total_game_time: float
    time_increment: float
    rounds: int
    scoring: Dict[str, float] = None

    def __post_init__(self):
        if self.scoring is None:
            self.scoring = {"win": 1.0, "draw": 0.5, "loss": 0.0}


@dataclass
class GameResult:
    """
    Results from a single chess game.

    :param white_name: Name of white player
    :type white_name: str
    :param black_name: Name of black player
    :type black_name: str
    :param game_id: Unique game identifier
    :type game_id: str
    :param timestamp: Game timestamp
    :type timestamp: str
    :param result: Game result ("1-0", "0-1", "1/2-1/2", "*")
    :type result: str
    :param winner: Winner ("white", "black", or None for draw)
    :type winner: Optional[str]
    :param termination: Termination reason
    :type termination: str
    :param moves: List of moves in UCI format
    :type moves: List[str]
    :param total_moves: Total number of moves
    :type total_moves: int
    :param duration: Total game time in seconds
    :type duration: float
    :param starting_fen: Starting position FEN
    :type starting_fen: str
    :param final_fen: Final position FEN
    :type final_fen: str
    :param white_time_left: White player's remaining time
    :type white_time_left: float
    :param black_time_left: Black player's remaining time
    :type black_time_left: float
    :param round_number: Tournament round number (optional)
    :type round_number: Optional[int]
    """

    # Game identification
    white_name: str
    black_name: str
    game_id: str
    timestamp: str

    # Game outcome
    result: str  # "1-0", "0-1", "1/2-1/2", "*"
    winner: Optional[str]  # "white", "black", or None for draw
    termination: str  # "checkmate", "stalemate", "timeout", etc.

    # Game details
    moves: List[str]  # List of moves in UCI format
    total_moves: int
    duration: float  # Total game time in seconds

    # Position information
    starting_fen: str
    final_fen: str

    # Time information
    white_time_left: float
    black_time_left: float

    # Tournament information (optional)
    round_number: Optional[int] = None

    def to_pgn(self, time_limit: float = None, time_increment: float = None) -> str:
        """
        Convert game result to PGN format.

        :param time_limit: Time control limit
        :type time_limit: float
        :param time_increment: Time control increment
        :type time_increment: float
        :return: PGN string representation
        :rtype: str
        """
        # Create PGN game
        game = chess.pgn.Game()

        # Set headers
        game.headers["Event"] = "Student Chess Tournament"
        game.headers["Date"] = self.timestamp.split("T")[0].replace("-", ".")
        game.headers["White"] = self.white_name
        game.headers["Black"] = self.black_name
        game.headers["Result"] = self.result

        # Set FEN header if using custom starting position
        if self.starting_fen != chess.STARTING_FEN:
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = self.starting_fen

        # Set proper time control format
        if time_limit is not None:
            if time_limit == float("inf"):
                game.headers["TimeControl"] = "-"  # Unlimited time
            elif time_increment and time_increment > 0:
                game.headers["TimeControl"] = f"{int(time_limit)}+{int(time_increment)}"
            else:
                game.headers["TimeControl"] = str(int(time_limit))
        else:
            game.headers["TimeControl"] = "-"

        # Add moves
        # Set up the board with the starting position
        if self.starting_fen != chess.STARTING_FEN:
            board = chess.Board(self.starting_fen)
            game.setup(board)
        else:
            board = chess.Board()

        node = game

        for move_uci in self.moves:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    node = node.add_variation(move)
                    board.push(move)
            except ValueError:
                continue  # Skip invalid moves

        return str(game)
