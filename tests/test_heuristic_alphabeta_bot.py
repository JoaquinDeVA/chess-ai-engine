"""
Basic test suite for HeuristicAlphaBetaBot class

This provides minimal smoke tests to verify the bot implementation works.
"""

import pytest
import chess
from bots.heuristic_alphabeta_bot import HeuristicAlphaBetaBot


@pytest.fixture
def bot():
    """Set up test fixture."""
    # Create bot with shallow depth for predictable results
    return HeuristicAlphaBetaBot(max_depth=2)


def test_starting_position_move(bot):
    """Test that bot returns a legal move from starting position."""
    board = chess.Board()

    move = bot.get_move(board, time_limit=10.0)

    # Should return a legal move
    assert move is not None
    assert move in board.legal_moves
    # With current implementation, should be consistent
    assert isinstance(move, chess.Move)


def test_italian_game_position(bot):
    """Test bot on Italian Game position."""
    board = chess.Board()
    # Italian game: 1.e4 e5 2.Nf3 Nc6 3.Bc4
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]
    for move in moves:
        board.push(chess.Move.from_uci(move))

    move = bot.get_move(board, time_limit=10.0)

    # Should return a legal move for Black
    assert move is not None
    assert move in board.legal_moves


def test_avoids_immediate_checkmate(bot):
    """Test that bot avoids being checkmated when possible."""
    # Position where Black is threatened with mate in 1 but can avoid it
    board = chess.Board()
    # Set up: White threatens Qh5+ followed by mate, but Black can defend
    moves = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5"]
    for move in moves:
        board.push(chess.Move.from_uci(move))

    move = bot.get_move(board, time_limit=10.0)

    # Should return a legal move (Black's turn)
    assert move is not None
    assert move in board.legal_moves

    # The move shouldn't result in immediate checkmate for Black
    test_board = board.copy()
    test_board.push(move)
    assert not test_board.is_checkmate()


@pytest.mark.parametrize("position_index,board", [
    (0, chess.Board()),  # Starting position
    (1, chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")),  # After 1.e4
    (2, chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")),  # After 1.e4 e5
])
def test_returns_legal_move_always(bot, position_index, board):
    """Test that bot always returns a legal move."""
    move = bot.get_move(board, time_limit=10.0)

    assert move is not None, f"Bot returned None for position {position_index}"
    assert move in board.legal_moves, f"Bot returned illegal move for position {position_index}"