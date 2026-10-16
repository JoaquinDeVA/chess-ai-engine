"""Sparse board features for the linear evaluation models.

A position becomes a list of active feature indices and their values; a linear model scores it
as the sum of ``weights[index] * value`` over those entries. Features are tapered: the same piece
on the same square contributes to a middlegame block and to an endgame block, weighted by the
game phase.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, List, Tuple

import chess

#: Phase of the start position (sum of the phase weights of all pieces).
MAX_PHASE = 24
#: Size of one block: 6 piece types times 64 squares.
BLOCK_SIZE = 6 * 64
_MIDDLEGAME_BLOCK = 0
_ENDGAME_BLOCK = BLOCK_SIZE


def phase(board: chess.Board) -> float:
    """Game phase of a position: ``1`` in the opening, ``0`` in a bare-kings ending.

    :param board: The position.
    :return: ``min(24, minors + 2 * rooks + 4 * queens) / 24``, in ``[0, 1]``.
    """
    minors = rooks = queens = 0
    for piece in board.piece_map().values():
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            minors += 1
        elif piece.piece_type == chess.ROOK:
            rooks += 1
        elif piece.piece_type == chess.QUEEN:
            queens += 1
    return min(MAX_PHASE, minors + 2 * rooks + 4 * queens) / MAX_PHASE


class LinearFeatures(ABC):
    """Sparse encoding of a position for a linear model.

    :cvar name: Identifier stored in the saved model.
    """

    name: ClassVar[str]

    @abstractmethod
    def size(self, perspective: str) -> int:
        """Number of weights needed under ``perspective``.

        :param perspective: ``"white"`` or ``"side_to_move"``.
        :return: Weight count (without the bias).
        """

    @abstractmethod
    def active(self, board: chess.Board, perspective: str) -> Tuple[List[int], List[float]]:
        """Active feature indices and values of a position.

        :param board: The position.
        :param perspective: ``"white"`` or ``"side_to_move"``.
        :return: Parallel lists of indices and values.
        """


class PstFeatures(LinearFeatures):
    """Tapered piece-square tables: one weight per (piece type, square, phase block).

    Under ``"side_to_move"`` every piece of the player to move adds ``+1`` at its square seen from
    rank 1 (the board is mirrored by rank when Black moves), and every piece of the opponent adds
    ``-1`` at the square seen from its own home rank. The features are therefore antisymmetric:
    a position and its colour-mirror produce the same entries.

    Under ``"white"`` no mirroring is applied: White pieces add ``+1``, Black pieces ``-1``, and
    one extra entry ``+1`` (White to move) or ``-1`` (Black to move) is added at index 768.

    :cvar name: ``"pst_tapered"``.
    """

    name = "pst_tapered"
    _TURN_INDEX = 2 * BLOCK_SIZE

    def size(self, perspective: str) -> int:
        """Weights per block pair, plus one turn weight under ``"white"``.

        :param perspective: ``"white"`` or ``"side_to_move"``.
        :return: ``768`` or ``769``.
        """
        return 2 * BLOCK_SIZE + (1 if perspective == "white" else 0)

    def active(self, board: chess.Board, perspective: str) -> Tuple[List[int], List[float]]:
        """Index and value of every piece, in both phase blocks, plus the turn feature.

        :param board: The position.
        :param perspective: ``"white"`` or ``"side_to_move"``.
        :return: Two parallel lists; each piece gives two entries (middlegame, endgame).
        """
        tapered = phase(board)
        mover = board.turn
        indices: List[int] = []
        values: List[float] = []
        for square, piece in board.piece_map().items():
            type_offset = (piece.piece_type - 1) * 64
            oriented = square
            if perspective == "side_to_move" and piece.color == chess.BLACK:
                # Each colour is seen from its own home rank.
                oriented = chess.square_mirror(square)
            if perspective == "white":
                sign = 1.0 if piece.color == chess.WHITE else -1.0
            else:
                sign = 1.0 if piece.color == mover else -1.0
            index = type_offset + oriented
            indices.extend((_MIDDLEGAME_BLOCK + index, _ENDGAME_BLOCK + index))
            values.extend((sign * tapered, sign * (1.0 - tapered)))
        if perspective == "white":
            indices.append(self._TURN_INDEX)
            values.append(1.0 if mover == chess.WHITE else -1.0)
        return indices, values


class MaterialCountFeatures(LinearFeatures):
    """Piece-count differences: one weight per piece type (P, N, B, R, Q), no phase split.

    Under ``"side_to_move"`` each piece type contributes its count of the player to move minus
    the opponent's count. Under ``"white"`` the count is White's minus Black's, and one extra
    entry ``+1`` (White to move) or ``-1`` (Black to move) is added at index 5.

    :cvar name: ``"material5"``.
    """

    name = "material5"
    _TURN_INDEX = 5

    def size(self, perspective: str) -> int:
        """Five counts, plus one turn weight under ``"white"``.

        :param perspective: ``"white"`` or ``"side_to_move"``.
        :return: ``5`` or ``6``.
        """
        return 5 + (1 if perspective == "white" else 0)

    def active(self, board: chess.Board, perspective: str) -> Tuple[List[int], List[float]]:
        """One entry per piece type with its signed count, plus the turn feature.

        :param board: The position.
        :param perspective: ``"white"`` or ``"side_to_move"``.
        :return: Two parallel lists of length ``5`` (or ``6`` under ``"white"``).
        """
        mover = board.turn
        counts = [0.0] * 5
        for piece in board.piece_map().values():
            if piece.piece_type == chess.KING:
                continue
            if perspective == "white":
                sign = 1.0 if piece.color == chess.WHITE else -1.0
            else:
                sign = 1.0 if piece.color == mover else -1.0
            counts[piece.piece_type - 1] += sign
        indices = list(range(5))
        if perspective == "white":
            indices.append(self._TURN_INDEX)
            counts.append(1.0 if mover == chess.WHITE else -1.0)
        return indices, counts
