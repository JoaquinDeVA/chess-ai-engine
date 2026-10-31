"""Play one game between two bots with time control: results, clocks and forfeits.

Based on the AI course framework (Inteligencia Artificial e Ingeniería del Conocimiento,
CEU San Pablo University).
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import chess

from chess_engine.bots.search_bot import SearchBot

#: Terminations in which a player loses without the game ending on the board.
FORFEIT_TERMINATIONS = ("time_forfeit", "illegal_move", "player_error")


@dataclass
class GameResult:
    """
    Results from a single chess game.

    :param white_name: Name of white player
    :param black_name: Name of black player
    :param game_id: Unique game identifier
    :param timestamp: Game timestamp
    :param result: Game result ("1-0", "0-1", "1/2-1/2", "*")
    :param winner: Winner ("white", "black", or None for draw)
    :param termination: Termination reason ("checkmate", "stalemate", "time_forfeit", etc.)
    :param moves: List of moves in UCI format
    :param total_moves: Total number of moves
    :param duration: Total game time in seconds
    :param starting_fen: Starting position FEN
    :param final_fen: Final position FEN
    :param white_time_left: White player's remaining time
    :param black_time_left: Black player's remaining time
    :param round_number: Tournament round number (optional)
    """

    white_name: str
    black_name: str
    game_id: str
    timestamp: str
    result: str
    winner: Optional[str]
    termination: str
    moves: List[str]
    total_moves: int
    duration: float
    starting_fen: str
    final_fen: str
    white_time_left: float
    black_time_left: float
    round_number: Optional[int] = None


def play_game(
    white_player: SearchBot,
    black_player: SearchBot,
    time_limit: float = 300.0,
    time_increment: float = 0.0,
    starting_position: Optional[str] = None,
    verbose: bool = False,
    round_number: Optional[int] = None,
) -> GameResult:
    """
    Play a single game between any two players.

    Handles time limits, tracks time for each player, and enforces game rules.

    :param white_player: Player to play as white
    :type white_player: SearchBot
    :param black_player: Player to play as black
    :type black_player: SearchBot
    :param time_limit: Total time per player in seconds
    :type time_limit: float
    :param time_increment: Time increment per move in seconds
    :type time_increment: float
    :param starting_position: FEN string for custom starting position
    :type starting_position: Optional[str]
    :param verbose: Enable detailed output
    :type verbose: bool
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
            current_time_left = white_time_left if board.turn == chess.WHITE else black_time_left

            if verbose:
                color_name = "White" if board.turn == chess.WHITE else "Black"
                print(f"\n{color_name} ({current_player.name}) to move...")
                print(f"Time left: {current_time_left:.1f}s")

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

        return result

    except KeyboardInterrupt:
        if verbose:
            print("\nGame cancelled by user")
        raise
    except Exception as e:
        if verbose:
            print(f"Game error: {e}")
        raise
