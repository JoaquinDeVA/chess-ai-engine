"""Sparse board features for the network inputs, seen from the side to move.

A position becomes a list of active feature indices and their values; the first layer of the
network adds ``weights[index] * value`` over those entries.
"""

from typing import Dict, List, Tuple

import chess

#: Material value of each piece type for the capture gains; the king outweighs any gain.
_ROLE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}
#: Number of threat inputs.
THREAT_COUNT = 5
_GAIN_SCALE = 9.0
_EN_PRISE_SCALE = 4.0


def _en_prise(board: chess.Board, attacker: chess.Color) -> Tuple[int, int]:
    """What ``attacker`` can win at once on the board as it stands.

    A piece (kings aside) of the other side that ``attacker`` attacks gains its value when no
    piece defends it, else its value minus that of the cheapest attacker when that is positive.
    Attacks are those of the board as it stands: no x-rays, pins or en passant.

    :param board: The position.
    :param attacker: The side that captures.
    :return: ``(best gain, number of pieces it wins something on)``.
    """
    best = count = 0
    for square in chess.scan_forward(board.occupied_co[not attacker] & ~board.kings):
        attackers = board.attackers_mask(attacker, square)
        if not attackers:
            continue
        target = _ROLE_VALUE[board.piece_type_at(square)]
        if board.attackers_mask(not attacker, square):
            cheapest = min(
                _ROLE_VALUE[board.piece_type_at(s)] for s in chess.scan_forward(attackers)
            )
            gain = target - cheapest
        else:
            gain = target
        if gain > 0:
            best = max(best, gain)
            count += 1
    return best, count


def _threat_entries(board: chess.Board, base: int) -> Tuple[List[int], List[float]]:
    """The five threat inputs, from ``base`` on.

    In order: the best capture of the side to move, the best capture of its rival (both over
    ``9``), the number of rival pieces the mover wins something on, the number of the mover's
    pieces the rival does (both over ``4``), and ``1`` when the mover is in check.

    :param board: The position.
    :param base: Index of the first threat entry.
    :return: Two parallel lists of :data:`THREAT_COUNT` entries.
    """
    mover_gain, mover_count = _en_prise(board, board.turn)
    rival_gain, rival_count = _en_prise(board, not board.turn)
    values = [
        mover_gain / _GAIN_SCALE,
        rival_gain / _GAIN_SCALE,
        mover_count / _EN_PRISE_SCALE,
        rival_count / _EN_PRISE_SCALE,
        1.0 if board.is_check() else 0.0,
    ]
    return list(range(base, base + THREAT_COUNT)), values


#: Offset of the opponent's piece block in :class:`SlottedPstFeatures`.
OPPONENT_PIECE_BLOCK = 6 * 64


class SlottedPstFeatures:
    """One-hot piece-square features with one entry per piece.

    A piece of the player to move gets index ``type * 64 + square`` and a piece of the opponent
    gets ``384 + type * 64 + square``, the square seen from the home rank of its colour (mirrored
    by rank for Black). Every value is ``1.0``, so the vector has at most :attr:`max_active`
    non-zero entries.

    :cvar name: Identifier stored in the saved model, ``"pst_slotted"``.
    :cvar max_active: Most entries (one per piece, 32 at most).
    """

    name = "pst_slotted"
    max_active = 32

    def max_entries(self) -> int:
        """Most non-zero entries a position can have.

        :return: ``32`` (one per piece).
        """
        return self.max_active

    def size(self) -> int:
        """Two blocks of 384 piece-square inputs.

        :return: ``768``.
        """
        return 2 * OPPONENT_PIECE_BLOCK

    def active(self, board: chess.Board) -> Tuple[List[int], List[float]]:
        """Index of every piece, all values ``1.0``.

        :param board: The position.
        :return: Two parallel lists, one entry per piece.
        """
        mover = board.turn
        indices: List[int] = []
        for square, piece in board.piece_map().items():
            offset = (piece.piece_type - 1) * 64
            if piece.color != mover:
                offset += OPPONENT_PIECE_BLOCK
            oriented = chess.square_mirror(square) if piece.color == chess.BLACK else square
            indices.append(offset + oriented)
        return indices, [1.0] * len(indices)


class ThreatsPstFeatures(SlottedPstFeatures):
    """Piece squares plus five threat inputs.

    The threats are the best capture of each side, the pieces each side has en prise and the
    check flag (:func:`_threat_entries`); they stand for what a search of captures would see.
    They are dense inputs of the mover: the rival's view has none.

    :cvar name: ``"pst_threats"``.
    """

    name = "pst_threats"

    def max_entries(self) -> int:
        """Pieces and the threat entries.

        :return: ``32 + 5``.
        """
        return self.max_active + THREAT_COUNT

    def size(self) -> int:
        """The piece squares and the threat entries.

        :return: ``768 + 5``.
        """
        return super().size() + THREAT_COUNT

    def active(self, board: chess.Board) -> Tuple[List[int], List[float]]:
        """The piece entries, then the threat entries.

        :param board: The position.
        :return: Two parallel lists.
        """
        indices, values = super().active(board)
        dense_indices, dense_values = _threat_entries(board, super().size())
        return indices + dense_indices, values + dense_values


#: Largest count of each piece type that has its own material entry (pawn, knight, bishop,
#: rook, queen); a bigger count shares the entry of the largest.
MATERIAL_CAP = (8, 2, 2, 2, 1)
#: Material entries of one side, and of both.
MATERIAL_PER_SIDE = sum(MATERIAL_CAP)
#: Pawn-file entries of one side (one for a single pawn, one for two or more, per file).
PAWN_FILES_PER_SIDE = 16
#: Kinds of king bucket of one side.
KING_BUCKETS = 4
#: First index of each block of :class:`CtxPstFeatures`: the pieces come first, then the
#: material entries, the pawn files and the king buckets (the mover's, then the rival's).
MATERIAL_BASE = 768
#: Piece-square inputs of one side: the same 768 in every feature set, before the other families.
PIECE_INPUTS = MATERIAL_BASE
PAWN_FILES_BASE = MATERIAL_BASE + 2 * MATERIAL_PER_SIDE
KING_BASE = PAWN_FILES_BASE + 2 * PAWN_FILES_PER_SIDE


def king_bucket(king_square: int) -> int:
    """Bucket of a king by its square seen from its own home rank (``a1 = 0``).

    :param king_square: Square of the king, mirrored by rank for Black.
    :return: ``0`` castled short (f1-h1), ``1`` castled long (a1-c1), ``2`` at home (d1, e1),
        ``3`` anywhere else.
    """
    if king_square < 8:
        file = king_square
        return 0 if file >= 5 else 1 if file <= 2 else 2
    return 3


class CtxPstFeatures(SlottedPstFeatures):
    """Piece squares plus material, pawn files and king buckets, all one-hot.

    After the 768 piece entries come, for the mover and then for its rival, one entry per piece
    type present with its count (:data:`MATERIAL_CAP`), one entry per pawn file with a pawn (a
    second one when it has two or more) and the bucket of its king (:func:`king_bucket`). The
    rival's view of a position is the same entries with the two sides swapped, so the network
    reads each side's family from its own block. The model decides which families it uses.

    :cvar name: ``"pst_ctx"``.
    """

    name = "pst_ctx"

    def max_entries(self) -> int:
        """Pieces, up to five material entries and sixteen pawn-file entries, two buckets.

        :return: ``32 + 10 + 16 + 2``.
        """
        return self.max_active + 2 * len(MATERIAL_CAP) + 2 * 8 + 2

    def size(self) -> int:
        """Piece squares, material, pawn files and the king buckets of both sides.

        :return: ``768 + 30 + 32 + 8``.
        """
        return KING_BASE + 2 * KING_BUCKETS

    def active(self, board: chess.Board) -> Tuple[List[int], List[float]]:
        """The piece entries, then material, pawn files and king buckets.

        :param board: The position.
        :return: Two parallel lists.
        """
        indices, _ = super().active(board)
        mover = board.turn
        buckets = []
        for side, color in enumerate((mover, not mover)):
            for position, piece_type in enumerate(
                (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
            ):
                count = min(len(board.pieces(piece_type, color)), MATERIAL_CAP[position])
                if count:
                    first = sum(MATERIAL_CAP[:position])
                    indices.append(MATERIAL_BASE + side * MATERIAL_PER_SIDE + first + count - 1)
            per_file = [0] * 8
            for square in board.pieces(chess.PAWN, color):
                per_file[chess.square_file(square)] += 1
            for file, count in enumerate(per_file):
                if count:
                    indices.append(
                        PAWN_FILES_BASE + side * PAWN_FILES_PER_SIDE + 2 * file + (count > 1)
                    )
            king = board.king(color)
            oriented = chess.square_mirror(king) if color == chess.BLACK else king
            buckets.append(KING_BASE + side * KING_BUCKETS + king_bucket(oriented))
        indices += buckets
        return indices, [1.0] * len(indices)


#: Every feature set by its name, the one the saved models and the core refer to.
FEATURE_SETS: Dict[str, SlottedPstFeatures] = {
    features.name: features
    for features in (SlottedPstFeatures(), ThreatsPstFeatures(), CtxPstFeatures())
}


def features_of(model: object) -> SlottedPstFeatures:
    """Python feature set of a saved network, or of its class.

    :param model: A saved network or its class.
    :return: The feature set named by its ``feature_set``.
    :raises TypeError: If it is the abstract base, which defines no feature set.
    :raises KeyError: If the name has no Python version.
    """
    name = getattr(model, "feature_set", None)
    if name is None:
        raise TypeError(f"{model!r} defines no feature set; use a concrete model")
    return FEATURE_SETS[name]
