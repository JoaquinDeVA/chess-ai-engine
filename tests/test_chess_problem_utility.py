"""
Test suite for ChessProblem.utility() function

This test file helps students verify their understanding of the utility function
and provides examples of terminal chess positions for different game outcomes.
"""

import pytest
import chess
from adversarial_search.chess_problem import ChessProblem
from adversarial_search.chess_game_state import ChessGameState


@pytest.fixture
def problem():
    """Set up test fixture."""
    return ChessProblem()


def test_white_checkmate_win(problem):
    """Test utility when White wins by checkmate."""
    # Scholar's mate - White wins in 4 moves
    # 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7# 1-0
    board = chess.Board()
    moves = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]

    for move in moves:
        board.push(chess.Move.from_uci(move))

    # Verify it's checkmate
    assert board.is_checkmate()
    assert board.outcome().winner == chess.WHITE

    state = ChessGameState(board)

    # White should get utility 1.0 (win)
    assert problem.utility(state, chess.WHITE) == 1.0

    # Black should get utility 0.0 (loss)
    assert problem.utility(state, chess.BLACK) == 0.0


def test_black_checkmate_win(problem):
    """Test utility when Black wins by checkmate."""
    # Fool's mate - Black wins in 2 moves
    # 1.f3 e5 2.g4 Qh4# 0-1
    board = chess.Board()
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]

    for move in moves:
        board.push(chess.Move.from_uci(move))

    # Verify it's checkmate
    assert board.is_checkmate()
    assert board.outcome().winner == chess.BLACK

    state = ChessGameState(board)

    # Black should get utility 1.0 (win)
    assert problem.utility(state, chess.BLACK) == 1.0

    # White should get utility 0.0 (loss)
    assert problem.utility(state, chess.WHITE) == 0.0


def test_stalemate_draw(problem):
    """Test utility for stalemate (draw)."""
    # Classic stalemate position: White King on a8, Black King on a6, White Queen on b6
    # Black to move, no legal moves, not in check = stalemate
    board = chess.Board()
    board.clear()
    board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.C7, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.B6, chess.Piece(chess.QUEEN, chess.WHITE))
    board.turn = chess.BLACK

    # Verify it's stalemate
    assert board.is_stalemate()
    assert board.outcome().winner is None

    state = ChessGameState(board)

    # Both players should get utility 0.5 (draw)
    assert problem.utility(state, chess.WHITE) == 0.5
    assert problem.utility(state, chess.BLACK) == 0.5


def test_insufficient_material_draw(problem):
    """Test utility for insufficient material draw."""
    # King vs King - insufficient material
    board = chess.Board()
    board.clear()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))

    # Verify it's insufficient material
    assert board.is_insufficient_material()
    assert board.outcome().winner is None

    state = ChessGameState(board)

    # Both players should get utility 0.5 (draw)
    assert problem.utility(state, chess.WHITE) == 0.5
    assert problem.utility(state, chess.BLACK) == 0.5


def test_threefold_repetition_draw(problem):
    """Test utility for threefold repetition draw."""
    board = chess.Board()

    # Create a position that will repeat 3 times
    # Simple knight moves that repeat the position
    moves = [
        "g1f3", "g8f6",  # Knights out
        "f3g1", "f6g8",  # Knights back (first repetition)
        "g1f3", "g8f6",  # Knights out again
        "f3g1", "f6g8"   # Knights back (second repetition)
    ]

    for move in moves:
        board.push(chess.Move.from_uci(move))

    # Push the moves one more time to trigger threefold repetition
    board.push(chess.Move.from_uci("g1f3"))
    board.push(chess.Move.from_uci("g8f6"))
    board.push(chess.Move.from_uci("f3g1"))
    board.push(chess.Move.from_uci("f6g8"))

    # Check if we can claim a draw (the game might not automatically end)
    state = ChessGameState(board)

    # Note: This test might need adjustment based on how ChessGameState handles repetition
    # If the state doesn't automatically detect repetition, this test documents the expected behavior
    assert state.is_over()
    # Both players should get utility 0.5 (draw)
    assert problem.utility(state, chess.WHITE) == 0.5
    assert problem.utility(state, chess.BLACK) == 0.5


def test_value_error_non_terminal_state(problem):
    """Test that ValueError is raised for non-terminal positions."""
    # Starting position - game is not over
    board = chess.Board()
    state = ChessGameState(board)

    # Should raise ValueError since game is not over
    with pytest.raises(ValueError) as exc_info:
        problem.utility(state, chess.WHITE)

    assert "terminal" in str(exc_info.value).lower()


def test_value_error_wrong_state_type(problem):
    """Test that ValueError is raised for wrong state type."""
    # Pass a non-ChessGameState object
    invalid_state = "not a chess state"

    # Should raise ValueError for wrong type
    with pytest.raises(ValueError) as exc_info:
        problem.utility(invalid_state, chess.WHITE)

    assert "ChessGameState" in str(exc_info.value)


def test_value_error_none_state(problem):
    """Test that ValueError is raised for None state."""
    # Should raise ValueError for None
    with pytest.raises(ValueError) as exc_info:
        problem.utility(None, chess.WHITE)

    assert "ChessGameState" in str(exc_info.value)



#Probar que el argumento jugador sea tambien válido

def test_invalid_player(problem):
    """Test that ValueError is raised for invalid player input."""

    board = chess.Board()
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]  # Fool's mate
    for move in moves:
        board.push(chess.Move.from_uci(move))

    state = ChessGameState(board)

    # Diferentes jugadores inválidos a probar
    invalid_players = [None, "white", 2, -1, 3.14, object()]

    for invalid_player in invalid_players:
        with pytest.raises(ValueError) as exc_info:
            problem.utility(state, invalid_player)

        # Mirar que el valueError es sobre el jugador
        assert "player" in str(exc_info.value).lower()