"""Hand-crafted position features: material and mobility."""

import chess

from chess_engine.evaluation.features import extract_material_features, extract_mobility_features


def test_starting_position_material():
    """Test material features for starting position (equal material)."""
    board = chess.Board()
    features = extract_material_features(board)

    # Both sides should have equal material (39 points each)
    assert features["white_material"] == 39
    assert features["black_material"] == 39
    assert features["material_diff"] == 0
    assert features["material_diff_normalized"] == 0.5


def test_material_advantage_white():
    """Test material features when White has material advantage."""
    # Remove black queen (9 points advantage for white)
    board = chess.Board()
    board.remove_piece_at(chess.D8)  # Remove black queen

    features = extract_material_features(board)

    assert features["white_material"] == 39
    assert features["black_material"] == 30  # 39 - 9 = 30
    assert features["material_diff"] == 9
    # Normalized: (9 + 39) / 78 = 48/78 ≈ 0.615
    assert abs(features["material_diff_normalized"] - 48 / 78) < 0.001


def test_starting_position_mobility():
    """Test mobility features for starting position."""
    board = chess.Board()
    features = extract_mobility_features(board)

    # Both sides should have equal mobility (20 moves each in starting position)
    assert features["white_mobility"] == 20
    assert features["black_mobility"] == 20
    assert features["mobility_diff"] == 0
    assert features["mobility_diff_normalized"] == 0.5


def test_restricted_mobility():
    """Test mobility when one side has restricted movement."""
    # Create a position where black has fewer moves
    board = chess.Board()
    # Play some moves to create asymmetric mobility
    moves = ["e2e4", "a7a6", "d1h5", "b7b6"]  # White develops queen aggressively
    for move in moves:
        board.push(chess.Move.from_uci(move))

    features = extract_mobility_features(board)

    # White should have more mobility due to active queen
    assert features["white_mobility"] > features["black_mobility"]
    assert features["mobility_diff"] > 0
    assert features["mobility_diff_normalized"] > 0.5


def test_specific_mobility_values():
    """Test mobility with specific known values."""
    # After 1.e4 e5 2.Nf3, black to move
    board = chess.Board()
    moves = ["e2e4", "e7e5", "g1f3"]
    for move in moves:
        board.push(chess.Move.from_uci(move))

    features = extract_mobility_features(board)

    # White should have 27 legal moves, black should have 29
    assert features["white_mobility"] == 27
    assert features["black_mobility"] == 29
    assert features["mobility_diff"] == -2
    # Normalized: (-2 + 80) / 160 = 78/160 = 0.4875
    assert abs(features["mobility_diff_normalized"] - 78 / 160) < 0.001


def test_endgame_mobility():
    """Test mobility in a simple endgame position."""
    # King and queen vs lone king
    board = chess.Board()
    board.clear()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE

    features = extract_mobility_features(board)

    # White has 21 moves, black king has 3 moves
    assert features["white_mobility"] == 21
    assert features["black_mobility"] == 3
    assert features["mobility_diff"] == 18
    # Normalized: (18 + 80) / 160 = 98/160 = 0.6125
    assert abs(features["mobility_diff_normalized"] - 98 / 160) < 0.001
