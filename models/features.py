"""Board encodings (feature extractors) used as model inputs.

A :class:`FeatureExtractor` turns a :class:`chess.Board` into one or more numeric arrays. Models
declare which extractor they use, so the same extractor is applied when training and when
playing.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Iterable, Tuple

import chess
import numpy as np

#: Material value of each piece type (kings excluded).
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

#: Material of one full side without the king.
MAX_SIDE_MATERIAL = 39.0
_CENTER_SQUARES = frozenset({chess.D4, chess.D5, chess.E4, chess.E5})
_FULL_PIECE_COUNTS = np.array([8, 2, 2, 2, 1], dtype=np.float32)


class FeatureExtractor(ABC):
    """Turns boards into model inputs.

    :cvar name: Short identifier stored in model cards.
    :cvar input_shapes: Shape of every array returned by :meth:`encode` (without batch axis).
    """

    name: ClassVar[str]
    input_shapes: ClassVar[Tuple[Tuple[int, ...], ...]]

    @abstractmethod
    def encode(self, board: chess.Board) -> Tuple[np.ndarray, ...]:
        """Encode one position.

        :param board: Position to encode.
        :return: One array per model input, without batch axis.
        """

    def encode_batch(self, boards: Iterable[chess.Board]) -> Tuple[np.ndarray, ...]:
        """Encode many positions and stack each input along a new first axis.

        :param boards: Positions to encode.
        :return: One array per model input with shape ``(n, *input_shape)``.
        """
        encoded = [self.encode(board) for board in boards]
        if not encoded:
            return tuple(np.empty((0, *shape), dtype=np.float32) for shape in self.input_shapes)
        return tuple(np.stack(column) for column in zip(*encoded))

    def encode_fens(self, fens: Iterable[str]) -> Tuple[np.ndarray, ...]:
        """Encode positions given as FEN strings.

        :param fens: FEN strings (the move counters may be missing).
        :return: Same as :meth:`encode_batch`.
        """
        return self.encode_batch(chess.Board(fen) for fen in fens)


class VectorFeatures(FeatureExtractor):
    """29 hand-crafted numbers: turn, phase, material, piece counts, pawns, centre, kings, ...

    Layout: ``turn, phase, 1-phase, material[0], material[1], material[0]-material[1],
    piece counts[0] (5), piece counts[1] (5), pawn advance (2), pawn count (2), centre
    occupation (2), king distance to the centre (2), in-check flags White/Black (2), king-side
    castling rights White/Black (2), mobility``.

    .. note::
       Every per-side pair is indexed with ``int(piece.color)``, i.e. **index 0 is Black and
       index 1 is White** (``chess.WHITE == True``). The shipped models were trained with this
       exact layout, so it is kept as is; only the in-check and castling pairs are ordered
       White, Black.
    """

    name = "vector29"
    input_shapes = ((29,),)

    def encode(self, board: chess.Board) -> Tuple[np.ndarray, ...]:
        """Encode one position into the 29-dimensional vector (see the class docstring).

        :param board: Position to encode.
        :return: A 1-tuple holding a ``float32`` array of shape ``(29,)``.
        """
        material = [0.0, 0.0]
        piece_count = np.zeros((2, 5), dtype=np.float32)
        pawn_rank_sum = [0.0, 0.0]
        pawn_count = [0, 0]
        center_control = [0, 0]
        king_square = [None, None]

        for square, piece in board.piece_map().items():
            color = int(piece.color)
            piece_type = piece.piece_type

            if piece_type != chess.KING:
                material[color] += PIECE_VALUES[piece_type]
                piece_count[color, piece_type - 1] += 1
            else:
                king_square[color] = square

            if piece_type == chess.PAWN:
                pawn_count[color] += 1
                pawn_rank_sum[color] += chess.square_rank(square) / 7.0

            if square in _CENTER_SQUARES:
                center_control[color] += 1

        material_diff = material[0] - material[1]
        phase = (material[0] + material[1]) / (2 * MAX_SIDE_MATERIAL)
        pawn_rank_avg = [
            pawn_rank_sum[c] / pawn_count[c] if pawn_count[c] > 0 else 0.0 for c in (0, 1)
        ]

        def king_distance_to_center(square):
            if square is None:
                return 0.0
            file_dist = abs(chess.square_file(square) - 3.5)
            rank_dist = abs(chess.square_rank(square) - 3.5)
            return (file_dist + rank_dist) / 7.0

        vector = np.array(
            [
                1.0 if board.turn == chess.WHITE else -1.0,
                phase,
                1.0 - phase,
                material[0] / MAX_SIDE_MATERIAL,
                material[1] / MAX_SIDE_MATERIAL,
                material_diff / MAX_SIDE_MATERIAL,
                *(piece_count[0] / _FULL_PIECE_COUNTS),
                *(piece_count[1] / _FULL_PIECE_COUNTS),
                pawn_rank_avg[0],
                pawn_rank_avg[1],
                pawn_count[0] / 8.0,
                pawn_count[1] / 8.0,
                center_control[0] / 4.0,
                center_control[1] / 4.0,
                king_distance_to_center(king_square[0]),
                king_distance_to_center(king_square[1]),
                board.is_check() and board.turn == chess.WHITE,
                board.is_check() and board.turn == chess.BLACK,
                board.has_kingside_castling_rights(chess.WHITE),
                board.has_kingside_castling_rights(chess.BLACK),
                board.legal_moves.count() / 50.0,
            ],
            dtype=np.float32,
        )
        return (vector,)


class BitboardFeatures(FeatureExtractor):
    """Two inputs for convolutional networks: an ``8x8x12`` piece tensor and 9 extra numbers.

    The tensor has one plane per piece type and colour (White P N B R Q K, then Black), with
    rank 8 in row 0. The 9 extra numbers are: turn, the four castling rights, both sides'
    material (divided by 39) and the in-check flags of White and Black.
    """

    name = "bitboard8x8x12+9"
    input_shapes = ((8, 8, 12), (9,))

    def encode(self, board: chess.Board) -> Tuple[np.ndarray, ...]:
        """Encode one position into the piece tensor and the auxiliary vector.

        :param board: Position to encode.
        :return: ``(tensor int8 (8, 8, 12), extras float32 (9,))``.
        """
        tensor = np.zeros((8, 8, 12), dtype=np.int8)
        material = [0.0, 0.0]
        for square, piece in board.piece_map().items():
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            plane = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
            tensor[row, col, plane] = 1
            if piece.piece_type != chess.KING:
                material[int(piece.color)] += PIECE_VALUES[piece.piece_type]

        extras = np.array(
            [
                1.0 if board.turn == chess.WHITE else -1.0,
                board.has_kingside_castling_rights(chess.WHITE),
                board.has_queenside_castling_rights(chess.WHITE),
                board.has_kingside_castling_rights(chess.BLACK),
                board.has_queenside_castling_rights(chess.BLACK),
                material[1] / MAX_SIDE_MATERIAL,
                material[0] / MAX_SIDE_MATERIAL,
                board.is_check() and board.turn == chess.WHITE,
                board.is_check() and board.turn == chess.BLACK,
            ],
            dtype=np.float32,
        )
        return tensor, extras


class MaterialFeatures(FeatureExtractor):
    """Six numbers: side to move and the White-minus-Black count of each piece type."""

    name = "material6"
    input_shapes = ((6,),)

    def encode(self, board: chess.Board) -> Tuple[np.ndarray, ...]:
        """Encode one position into side-to-move plus piece-count differences.

        :param board: Position to encode.
        :return: A 1-tuple holding a ``float32`` array of shape ``(6,)``.
        """
        differences = [
            len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK))
            for piece_type in PIECE_VALUES
        ]
        turn = 1.0 if board.turn == chess.WHITE else -1.0
        return (np.array([turn, *differences], dtype=np.float32),)
