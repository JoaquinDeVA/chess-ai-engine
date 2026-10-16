"""
Data classes for the chess engine.

This module contains all the data structures used throughout the chess engine
(tournament configuration, game and tournament results, PGN export).

Based on the AI course framework (Inteligencia Artificial e Ingeniería del Conocimiento,
CEU San Pablo University).
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class TournamentResult:
    """
    Results from a complete tournament.

    :param tournament_id: Unique tournament identifier
    :type tournament_id: str
    :param timestamp: Tournament timestamp
    :type timestamp: str
    :param config: Tournament configuration
    :type config: TournamentConfig
    :param bot_names: List of participating bots
    :type bot_names: List[str]
    :param games: List of all games played
    :type games: List[GameResult]
    :param duration: Total tournament duration in seconds
    :type duration: float
    """

    tournament_id: str
    timestamp: str
    config: TournamentConfig
    bot_names: List[str]
    games: List[GameResult]
    duration: float

    def get_standings(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Calculate tournament standings.

        :return: List of (bot_name, stats) tuples sorted by score
        :rtype: List[Tuple[str, Dict[str, Any]]]
        """
        standings = {}

        # Initialize standings for all bots
        for bot_name in self.bot_names:
            standings[bot_name] = {
                "score": 0.0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "games_played": 0,
            }

        # Get scoring rules
        scoring = self.config.scoring

        # Process each game result
        for game in self.games:
            white_stats = standings[game.white_name]
            black_stats = standings[game.black_name]

            white_stats["games_played"] += 1
            black_stats["games_played"] += 1

            if game.winner == "white":
                white_stats["wins"] += 1
                white_stats["score"] += scoring["win"]
                black_stats["losses"] += 1
                black_stats["score"] += scoring["loss"]
            elif game.winner == "black":
                black_stats["wins"] += 1
                black_stats["score"] += scoring["win"]
                white_stats["losses"] += 1
                white_stats["score"] += scoring["loss"]
            else:  # Draw
                white_stats["draws"] += 1
                white_stats["score"] += scoring["draw"]
                black_stats["draws"] += 1
                black_stats["score"] += scoring["draw"]

        # Sort by score (descending), then by wins
        sorted_standings = sorted(
            standings.items(), key=lambda x: (x[1]["score"], x[1]["wins"]), reverse=True
        )

        return sorted_standings

    def save_to_file(self, filename: str) -> None:
        """
        Save tournament results to JSON file.

        :param filename: Output filename
        :type filename: str
        """
        data = asdict(self)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def export_games_to_pgn_directory(self, directory_path: str) -> None:
        """
        Export all tournament games to PGN files in a directory.

        :param directory_path: Directory path for PGN files
        :type directory_path: str
        """
        import os

        # Create directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)

        for game in self.games:
            # Create filename: white_player-black_player-round.pgn
            if game.round_number is not None:
                filename = f"{game.white_name}-{game.black_name}-{game.round_number}.pgn"
            else:
                filename = f"{game.white_name}-{game.black_name}.pgn"

            file_path = os.path.join(directory_path, filename)

            try:
                pgn_content = game.to_pgn(
                    time_limit=self.config.total_game_time,
                    time_increment=self.config.time_increment,
                )
                with open(file_path, "w") as f:
                    f.write(pgn_content)
            except Exception as e:
                print(f"Warning: Failed to export {filename}: {e}")
