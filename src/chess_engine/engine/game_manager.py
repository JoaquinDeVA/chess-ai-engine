"""
Game manager: chess game and tournament orchestration.

This module manages chess games, tournaments, and coordinates between
bots, time controls, and game rules.

Based on the AI course framework (Inteligencia Artificial e Ingeniería del Conocimiento,
CEU San Pablo University).
"""

import itertools
import json
import time
from datetime import datetime
from typing import List, Optional

import chess

from chess_engine.bots.bot_registry import BotRegistry
from chess_engine.bots.chess_bot import ChessBot
from chess_engine.bots.human_player import HumanPlayer
from chess_engine.config.settings import BLITZ_CONFIG
from chess_engine.engine.data_classes import GameResult, TournamentConfig, TournamentResult

#: Terminations in which a player loses without the game ending on the board.
FORFEIT_TERMINATIONS = ("time_forfeit", "illegal_move", "player_error")


class GameManager:
    """
    Manages chess games and tournaments.

    This class coordinates between bots, handles time controls,
    manages game state, and processes results. It's the central
    orchestrator for all chess activities.
    """

    def __init__(self, bot_registry: BotRegistry):
        """
        Initialize the game manager.

        :param bot_registry: Registry containing all available bots
        :type bot_registry: BotRegistry
        """
        self.bot_registry = bot_registry
        self.completed_games: List[GameResult] = []

    def _format_board_emoji(self, board: chess.Board) -> str:
        """
        Format chess board with emoji pieces.

        :param board: Chess board to format
        :type board: chess.Board
        :return: Formatted board string with emoji pieces
        :rtype: str
        """
        piece_symbols = {
            "r": "♜",
            "n": "♞",
            "b": "♝",
            "q": "♛",
            "k": "♚",
            "p": "♟",
            "R": "♖",
            "N": "♘",
            "B": "♗",
            "Q": "♕",
            "K": "♔",
            "P": "♙",
        }

        board_str = ""
        for rank in range(8, 0, -1):
            board_str += f"{rank} "
            for file in range(8):
                square = chess.square(file, rank - 1)
                piece = board.piece_at(square)
                if piece:
                    board_str += piece_symbols.get(piece.symbol(), piece.symbol())
                else:
                    board_str += "."
                board_str += "    "
            board_str += "\n"

        board_str += "  a    b    c    d    e    f    g    h\n"
        return board_str

    def _display_board(self, board: chess.Board, style: str = "ascii") -> None:
        """
        Display chess board in specified style.

        :param board: Chess board to display
        :type board: chess.Board
        :param style: Display style ("ascii" or "emoji")
        :type style: str
        """
        if style == "emoji":
            print(self._format_board_emoji(board))
        else:
            print(board)

    def play_game(
        self,
        white_player: ChessBot,
        black_player: ChessBot,
        time_limit: float = 300.0,
        time_increment: float = 0.0,
        starting_position: Optional[str] = None,
        verbose: bool = False,
        board_style: str = "ascii",
        round_number: Optional[int] = None,
    ) -> GameResult:
        """
        Play a single game between any two players.

        Supports any combination of players (human vs human, bot vs bot, human vs bot).
        Handles time limits, tracks time for each player, and enforces game rules.

        :param white_player: Player to play as white
        :type white_player: ChessBot
        :param black_player: Player to play as black
        :type black_player: ChessBot
        :param time_limit: Total time per player in seconds
        :type time_limit: float
        :param time_increment: Time increment per move in seconds
        :type time_increment: float
        :param starting_position: FEN string for custom starting position
        :type starting_position: Optional[str]
        :param verbose: Enable detailed output
        :type verbose: bool
        :param board_style: Display style ("ascii" or "emoji")
        :type board_style: str
        :param round_number: Round number for tournament games
        :type round_number: Optional[int]
        :return: GameResult with complete game information
        :rtype: GameResult
        """
        # Set up the game
        board = chess.Board() if starting_position is None else chess.Board(starting_position)
        starting_fen = board.fen()

        # Initialize time tracking
        white_time_left = time_limit
        black_time_left = time_limit

        if verbose:
            print(f"Starting game: {white_player.name} (White) vs {black_player.name} (Black)")
            print(f"Time control: {time_limit}s + {time_increment}s increment")
            if starting_position:
                print(f"Custom position: {starting_position}")

        # Notify players of game start
        white_player.on_game_start(chess.WHITE, black_player.name)
        black_player.on_game_start(chess.BLACK, white_player.name)

        # Play the game
        moves = []
        start_time = time.time()
        winner = None
        termination = "unknown"

        try:
            while not board.is_game_over(claim_draw=True):
                current_player = white_player if board.turn == chess.WHITE else black_player
                current_time_left = (
                    white_time_left if board.turn == chess.WHITE else black_time_left
                )

                if verbose:
                    color_name = "White" if board.turn == chess.WHITE else "Black"
                    print(f"\n{color_name} ({current_player.name}) to move...")
                    print(f"Time left: {current_time_left:.1f}s")

                if isinstance(current_player, HumanPlayer):
                    self._display_board(board, board_style)

                # Check if player has time left
                if current_time_left <= 0:
                    if verbose:
                        print(f"Time forfeit: {current_player.name} ran out of time")
                    winner = "black" if board.turn == chess.WHITE else "white"
                    termination = "time_forfeit"
                    break

                # Get move with time tracking
                move_start = time.time()
                try:
                    move = current_player.get_move(board, current_time_left)
                except Exception as e:
                    if verbose:
                        print(f"Error getting move from {current_player.name}: {e}")
                    # Treat as illegal move - player loses
                    winner = "black" if board.turn == chess.WHITE else "white"
                    termination = "player_error"
                    break

                move_time = time.time() - move_start

                # Update time left
                if board.turn == chess.WHITE:
                    white_time_left = max(0, white_time_left - move_time + time_increment)
                else:
                    black_time_left = max(0, black_time_left - move_time + time_increment)

                # Check for time forfeit after move
                if move_time > current_time_left:
                    if verbose:
                        print(f"Time forfeit: {current_player.name} exceeded time limit")
                    winner = "black" if board.turn == chess.WHITE else "white"
                    termination = "time_forfeit"
                    break

                # Validate move
                if move is None or move not in board.legal_moves:
                    if verbose:
                        print(f"Illegal move attempted by {current_player.name}: {move}")
                        print("Game over: Illegal move")
                    winner = "black" if board.turn == chess.WHITE else "white"
                    termination = "illegal_move"
                    break

                # Make the move (the algebraic notation must be computed before pushing it)
                move_desc = board.san(move) if verbose else move.uci()
                moves.append(move.uci())
                board.push(move)

                if verbose:
                    print(f"Move: {move_desc} (took {move_time:.2f}s)")

            # Game finished
            game_duration = time.time() - start_time

            # Determine final result if not already decided
            if winner is None:
                outcome = board.outcome(claim_draw=True)
                if outcome.winner is not None:
                    winner = "white" if outcome.winner == chess.WHITE else "black"
                termination = str(outcome.termination.name).lower()

            # Create game result
            if termination in FORFEIT_TERMINATIONS:
                game_result = "1-0" if winner == "white" else "0-1"
            else:
                game_result = board.result(claim_draw=True)

            result = GameResult(
                white_name=white_player.name,
                black_name=black_player.name,
                game_id=f"{white_player.name}_vs_{black_player.name}_{int(time.time())}",
                timestamp=datetime.now().isoformat(),
                result=game_result,
                winner=winner,
                termination=termination,
                moves=moves,
                total_moves=len(moves),
                duration=game_duration,
                starting_fen=starting_fen,
                final_fen=board.fen(),
                white_time_left=white_time_left,
                black_time_left=black_time_left,
                round_number=round_number,
            )

            # Notify players of game end
            white_player.on_game_end(result.result, board)
            black_player.on_game_end(result.result, board)

            self.completed_games.append(result)
            return result

        except KeyboardInterrupt:
            if verbose:
                print("\nGame cancelled by user")
            raise
        except Exception as e:
            if verbose:
                print(f"Game error: {e}")
            raise

    def run_tournament(
        self, bot_names: List[str], config: TournamentConfig, verbose: bool = False
    ) -> "TournamentResult":
        """
        Run a round-robin tournament between bots.

        Each bot plays every other bot twice (once as white, once as black)
        for the specified number of rounds.

        :param bot_names: List of bot names to include in tournament
        :type bot_names: List[str]
        :param config: Tournament configuration
        :type config: TournamentConfig
        :param verbose: Enable detailed output
        :type verbose: bool
        :return: TournamentResult with complete tournament information
        :rtype: TournamentResult
        """
        if len(bot_names) < 2:
            raise ValueError("Tournament requires at least 2 bots")

        tournament_start = time.time()
        tournament_id = f"tournament_{int(tournament_start)}"

        # Load all bots
        bots = {}
        for bot_name in bot_names:
            bot = self.bot_registry.get_bot(bot_name)
            if bot is None:
                raise ValueError(f"Bot '{bot_name}' not found or failed to load")
            bots[bot_name] = bot

        if verbose:
            print(f"Loaded {len(bots)} bots for tournament")
            print(f"Tournament format: {config.rounds} round(s), round-robin")
            print(f"Time control: {config.total_game_time}s + {config.time_increment}s")

        all_games = []
        game_number = 0

        # Run the specified number of rounds
        for round_num in range(1, config.rounds + 1):
            if verbose:
                print(f"\n=== Round {round_num} ===")

            # Generate all pairings (each bot plays every other bot twice per round)
            for white_name, black_name in itertools.permutations(bot_names, 2):
                game_number += 1
                white_bot = bots[white_name]
                black_bot = bots[black_name]

                if verbose:
                    print(f"Game {game_number}: {white_name} (White) vs {black_name} (Black)")

                # Play the game using the improved play_game method
                # All error handling is done in play_game - it will return a proper GameResult
                game_result = self.play_game(
                    white_player=white_bot,
                    black_player=black_bot,
                    time_limit=config.total_game_time,
                    time_increment=config.time_increment,
                    starting_position=None,
                    verbose=False,  # Keep individual games quiet unless very verbose
                    round_number=round_num,
                )

                all_games.append(game_result)

                if verbose:
                    result_str = f"Result: {game_result.result}"
                    if game_result.winner:
                        result_str += f" ({game_result.winner} wins)"
                    else:
                        result_str += " (draw)"
                    result_str += f" - {game_result.total_moves} moves, {game_result.duration:.1f}s"
                    result_str += f" ({game_result.termination})"
                    print(f"  {result_str}")

        tournament_duration = time.time() - tournament_start

        # Create tournament result
        tournament_result = TournamentResult(
            tournament_id=tournament_id,
            timestamp=datetime.now().isoformat(),
            config=config,
            bot_names=bot_names,
            games=all_games,
            duration=tournament_duration,
        )

        if verbose:
            print(f"\nTournament completed in {tournament_duration:.1f} seconds")
            print(f"Total games played: {len(all_games)}")

        return tournament_result


def load_config(config_path: str) -> TournamentConfig:
    """
    Load simple tournament configuration.

    :param config_path: Path to JSON config file
    :type config_path: str
    :return: TournamentConfig object
    :rtype: TournamentConfig
    """
    try:
        with open(config_path) as f:
            data = json.load(f)
        # FIXED: Pass all data as keyword arguments
        return TournamentConfig(**data)
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using defaults")
        return TournamentConfig(**BLITZ_CONFIG)
    except Exception as e:
        print(f"Error loading config: {e}, using defaults")
        return TournamentConfig(**BLITZ_CONFIG)
