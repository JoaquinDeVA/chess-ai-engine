""" ""
Basic test suite for heuristic_alphabeta_search function

This provides minimal smoke tests to verify student implementations work.
Students should write more comprehensive tests themselves.
"""

import chess
import pytest

from chess_engine.evaluation.features import get_position_features
from chess_engine.search.chess_game_state import ChessGameState
from chess_engine.search.chess_problem import ChessProblem
from chess_engine.search.game_algorithms import heuristic_alphabeta_search


def simple_average_heuristic(state, player):
    """
    Simple test heuristic that averages all normalized features.

    This provides predictable behavior for testing purposes.
    """
    board = state.get_board()
    features = get_position_features(board)

    # Average all normalized features
    normalized_features = [
        features["material_diff_normalized"],
        features["positional_diff_normalized"],
        features["mobility_diff_normalized"],
        features["center_diff_normalized"],
        features["king_safety_diff_normalized"],
        features["development_diff_normalized"],
    ]

    avg = sum(normalized_features) / len(normalized_features)

    # Return from player's perspective
    if player == chess.WHITE:
        return avg
    else:
        return 1.0 - avg


def simple_cutoff(state, depth, elapsed_time):
    """Simple cutoff test for shallow search."""
    return depth >= 2


def test_starting_position_move():
    """Test that algorithm returns expected move from starting position."""
    board = chess.Board()
    problem = ChessProblem(board)
    state = ChessGameState(board)

    # 🔧 Modified line: now unpack move, stats
    move, stats = heuristic_alphabeta_search(
        problem, state, simple_average_heuristic, simple_cutoff
    )

    # With our simple heuristic and depth 2, should return e2e4
    assert move is not None
    assert move in board.legal_moves
    assert move.uci() == "e2e4"


def test_italian_game_position():
    """Test algorithm on Italian Game position."""
    board = chess.Board()
    # Italian game: 1.e4 e5 2.Nf3 Nc6 3.Bc4
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]
    for move in moves:
        board.push(chess.Move.from_uci(move))

    problem = ChessProblem(board)
    state = ChessGameState(board)

    # 🔧 Modified line: now unpack move, stats
    move, stats = heuristic_alphabeta_search(
        problem, state, simple_average_heuristic, simple_cutoff
    )

    # With our simple heuristic, should return d7d5 (Black's move)
    assert move is not None
    assert move in board.legal_moves
    assert move.uci() == "d7d5"


@pytest.mark.parametrize(
    "position_index,board",
    [
        (0, chess.Board()),  # Starting position
        (
            1,
            chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ),  # After 1.e4
        (
            2,
            chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"),
        ),  # After 1.e4 e5
    ],
)
def test_returns_legal_move_always(position_index, board):
    """Test that algorithm always returns a legal move."""
    problem = ChessProblem(board)
    state = ChessGameState(board)

    # 🔧 Modified line: now unpack move, stats
    move, stats = heuristic_alphabeta_search(
        problem, state, simple_average_heuristic, simple_cutoff
    )

    assert move is not None, f"Algorithm returned None for position {position_index}"
    assert move in board.legal_moves, (
        f"Algorithm returned illegal move for position {position_index}"
    )
