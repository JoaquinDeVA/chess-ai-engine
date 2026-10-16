"""
Chess Position Feature Extraction

This module provides functions to extract various features from chess positions
for use in evaluation functions.
"""
import copy
from typing import Dict


import chess

# Piece values for material evaluation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

# fmt: off
# Piece-square tables for positional evaluation (from white's perspective)
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10,-20,-20, 10, 10,  5,
    5, -5,-10,  0,  0,-10, -5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5,  5, 10, 25, 25, 10,  5,  5,
   10, 10, 20, 30, 30, 20, 10, 10,
   50, 50, 50, 50, 50, 50, 50, 50,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
  -50,-40,-30,-30,-30,-30,-40,-50,
  -40,-20,  0,  0,  0,  0,-20,-40,
  -30,  0, 10, 15, 15, 10,  0,-30,
  -30,  5, 15, 20, 20, 15,  5,-30,
  -30,  0, 15, 20, 20, 15,  0,-30,
  -30,  5, 10, 15, 15, 10,  5,-30,
  -40,-20,  0,  5,  5,  0,-20,-40,
  -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
  -20,-10,-10,-10,-10,-10,-10,-20,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -10,  0,  5, 10, 10,  5,  0,-10,
  -10,  5,  5, 10, 10,  5,  5,-10,
  -10,  0, 10, 10, 10, 10,  0,-10,
  -10, 10, 10, 10, 10, 10, 10,-10,
  -10,  5,  0,  0,  0,  0,  5,-10,
  -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_TABLE = [
   0,  0,  0,  0,  0,  0,  0,  0,
   5, 10, 10, 10, 10, 10, 10,  5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
   0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
  -20,-10,-10, -5, -5,-10,-10,-20,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -10,  0,  5,  5,  5,  5,  0,-10,
   -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
  -10,  5,  5,  5,  5,  5,  0,-10,
  -10,  0,  5,  0,  0,  0,  0,-10,
  -20,-10,-10, -5, -5,-10,-10,-20
]

KING_TABLE = [
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -20,-30,-30,-40,-40,-30,-30,-20,
  -10,-20,-20,-20,-20,-20,-20,-10,
   20, 20,  0,  0,  0,  0, 20, 20,
   20, 30, 10,  0,  0, 10, 30, 20
]
# fmt: on

PIECE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE,
}


def extract_material_features(board: chess.Board) -> Dict[str, float]:
    """
    Extract material-related features from the position.

    :param board: Chess board to analyze
    :type board: chess.Board
    :return: Dictionary containing material features
    :rtype: Dict[str, float]
    """

    if (board is None) or (not isinstance(board, chess.Board)):
        raise ValueError("board must be chess.Board")

    white_material = 0
    black_material = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is not None:
            if piece.color == chess.WHITE:
                white_material += PIECE_VALUES[piece.piece_type]
            else:
                black_material += PIECE_VALUES[piece.piece_type]

    material_diff = white_material - black_material

    return {
        "white_material": white_material,
        "black_material": black_material,
        "material_diff": material_diff,
        "material_diff_normalized": (material_diff + 39) / 78.0,
    }


def extract_mobility_features(board: chess.Board) -> Dict[str, float]:
    """
    Extract mobility features (number of legal moves for each side).

    :param board: Chess board to analyze
    :type board: chess.Board
    :return: Dictionary containing mobility features
    :rtype: Dict[str, float]
    """

    #Guardamos nuestro turno inicial
    turn = board.turn

    #Calculamos para blancas
    board.turn = chess.WHITE
    white_mobility = len(list(board.legal_moves))

    #Calculamos para negras
    board.turn = chess.BLACK
    black_mobility = len(list(board.legal_moves))

    #Restauramos el turno
    board.turn = turn

    mobility_diff =  white_mobility - black_mobility

    return {
        "white_mobility": white_mobility,
        "black_mobility": black_mobility,
        "mobility_diff": mobility_diff,
        "mobility_diff_normalized": (mobility_diff + 80) / 160.0,
    }


def extract_positional_features(board: chess.Board) -> Dict[str, float]:
    """
    Extract positional features using piece-square tables.

    :param board: Chess board to analyze
    :type board: chess.Board
    :return: Dictionary containing positional features
    :rtype: Dict[str, float]
    """
    white_positional = 0
    black_positional = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                table_index = square
            else:
                table_index = chess.square_mirror(square)

            positional_value = PIECE_TABLES[piece.piece_type][table_index] / 100.0

            if piece.color == chess.WHITE:
                white_positional += positional_value
            else:
                black_positional += positional_value

    positional_diff = white_positional - black_positional

    return {
        "white_positional": white_positional,
        "black_positional": black_positional,
        "positional_diff": positional_diff,
        "positional_diff_normalized": (positional_diff + 39) / 78.0,
    }


def extract_center_control_features(board: chess.Board) -> Dict[str, float]:
    """
    Extract center control features.

    :param board: Chess board to analyze
    :type board: chess.Board
    :return: Dictionary containing center control features
    :rtype: Dict[str, float]
    """
    center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]

    white_center = sum(
        1
        for sq in center_squares
        if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE
    )
    black_center = sum(
        1
        for sq in center_squares
        if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK
    )

    center_diff = white_center - black_center
    center_diff_clipped = max(-4, min(4, center_diff))
    return {
        "white_center": white_center,
        "black_center": black_center,
        "center_diff": center_diff,
        "center_diff_normalized": (center_diff_clipped + 4) / 8.0,
    }


def extract_king_safety_features(board: chess.Board) -> Dict[str, float]:
    """
    Extract king safety features.

    :param board: Chess board to analyze
    :type board: chess.Board
    :return: Dictionary containing king safety features
    :rtype: Dict[str, float]
    """
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)

    # King safety: prefer corners/edges in middlegame
    white_king_safety = (
        min(chess.square_file(white_king_sq), 7 - chess.square_file(white_king_sq))
        / 3.5
    )
    black_king_safety = (
        min(chess.square_file(black_king_sq), 7 - chess.square_file(black_king_sq))
        / 3.5
    )

    # Castling rights bonus
    white_king_safety += 0.5 if board.has_castling_rights(chess.WHITE) else 0
    black_king_safety += 0.5 if board.has_castling_rights(chess.BLACK) else 0

    king_safety_diff = white_king_safety - black_king_safety
    king_safety_diff_clipped = max(-4.0, min(4.0, king_safety_diff))
    return {
        "white_king_safety": white_king_safety,
        "black_king_safety": black_king_safety,
        "king_safety_diff": king_safety_diff,
        "king_safety_diff_normalized": (king_safety_diff_clipped + 4.0) / 8.0,
    }


def extract_development_features(board: chess.Board) -> Dict[str, float]:
    """
    Extract piece development features.

    :param board: Chess board to analyze
    :type board: chess.Board
    :return: Dictionary containing development features
    :rtype: Dict[str, float]
    """
    # Piece development (knights and bishops off back rank)
    white_dev = sum(
        1
        for sq, piece in board.piece_map().items()
        if (
            piece.color == chess.WHITE
            and piece.piece_type in [chess.KNIGHT, chess.BISHOP]
            and chess.square_rank(sq) > 0
        )
    )
    black_dev = sum(
        1
        for sq, piece in board.piece_map().items()
        if (
            piece.color == chess.BLACK
            and piece.piece_type in [chess.KNIGHT, chess.BISHOP]
            and chess.square_rank(sq) < 7
        )
    )

    development_diff = white_dev - black_dev
    development_diff_clipped = max(-4, min(4, development_diff))
    return {
        "white_development": white_dev,
        "black_development": black_dev,
        "development_diff": development_diff,
        "development_diff_normalized": (development_diff_clipped + 4) / 8.0,
    }


def extract_terminal_features(board: chess.Board) -> Dict[str, float]:
    """
    Extract terminal state features.

    :param board: Chess board to analyze
    :type board: chess.Board
    :return: Dictionary containing terminal state features
    :rtype: Dict[str, float]
    """
    if board.is_checkmate():
        return {"terminal": 1.0 if not board.turn else 0.0}
    if board.is_stalemate() or board.is_insufficient_material():
        return {"terminal": 0.5}

    return {"terminal": 0.0}


def get_position_features(board: chess.Board) -> Dict[str, float]:
    """
    Extract all normalized features from a chess position.

    :param board: Chess board to analyze
    :type board: chess.Board
    :return: Dictionary containing all normalized position features
    :rtype: Dict[str, float]
    """
    # Extract all feature categories
    features = {}
    features.update(extract_material_features(board))
    features.update(extract_positional_features(board))
    features.update(extract_mobility_features(board))
    features.update(extract_center_control_features(board))
    features.update(extract_king_safety_features(board))
    features.update(extract_development_features(board))

    return features

