"""Sparse board features for the linear evaluation models.

A position becomes a list of active feature indices and their values; a linear model scores it
as the sum of ``weights[index] * value`` over those entries. Features are tapered: the same piece
on the same square contributes to a middlegame block and to an endgame block, weighted by the
game phase.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Tuple

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


#: Names of the classical terms of :class:`PstTermsFeatures`, in weight order (two weights each).
TERM_NAMES = (
    "passed",
    "doubled",
    "isolated",
    "bishop_pair",
    "rook_open",
    "rook_half_open",
    "king_shield",
    "mobility_knight",
    "mobility_bishop",
    "mobility_rook",
    "mobility_queen",
)
_PAWN_CACHE_LIMIT = 100_000
#: Cache of the pawn-structure terms, keyed by the pawn bitboards of both colours.
_PAWN_CACHE: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}
_PAWN_SCALE = 4.0
_MOBILITY_SCALE = 10.0
_SHIELD_SCALE = 3.0
_MOBILITY_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def _color_pawn_counts(board: chess.Board, color: chess.Color) -> Tuple[int, int, int]:
    """Passed, doubled and isolated pawn counts of one colour.

    A pawn is passed when no enemy pawn stands ahead of it on its file or an adjacent file;
    doubled pawns are the pawns beyond the first on each file; isolated pawns have no friendly
    pawn on an adjacent file.

    :param board: The position.
    :param color: The colour whose pawns are counted.
    :return: ``(passed, doubled, isolated)``.
    """
    own = board.pawns & board.occupied_co[color]
    enemy = [
        (chess.square_file(square), chess.square_rank(square))
        for square in chess.scan_forward(board.pawns & board.occupied_co[not color])
    ]
    files = [0] * 8
    own_squares = list(chess.scan_forward(own))
    for square in own_squares:
        files[chess.square_file(square)] += 1
    doubled = sum(count - 1 for count in files if count > 1)
    passed = isolated = 0
    for square in own_squares:
        file, rank = chess.square_file(square), chess.square_rank(square)
        if not any(files[adjacent] for adjacent in (file - 1, file + 1) if 0 <= adjacent < 8):
            isolated += 1
        if color == chess.WHITE:
            blocked = any(abs(f - file) <= 1 and r > rank for f, r in enemy)
        else:
            blocked = any(abs(f - file) <= 1 and r < rank for f, r in enemy)
        if not blocked:
            passed += 1
    return passed, doubled, isolated


def _pawn_terms(board: chess.Board) -> Tuple[float, float, float]:
    """White-POV passed, doubled and isolated pawn terms, scaled by ``1 / 4``; cached.

    The cache holds at most :data:`_PAWN_CACHE_LIMIT` entries and is cleared when full.

    :param board: The position.
    :return: ``(passed, doubled, isolated)`` as White minus Black.
    """
    key = (
        board.pawns,
        board.occupied_co[chess.WHITE] & board.pawns,
        board.occupied_co[chess.BLACK] & board.pawns,
    )
    cached = _PAWN_CACHE.get(key)
    if cached is None:
        if len(_PAWN_CACHE) >= _PAWN_CACHE_LIMIT:
            _PAWN_CACHE.clear()
        white = _color_pawn_counts(board, chess.WHITE)
        black = _color_pawn_counts(board, chess.BLACK)
        cached = tuple((w - b) / _PAWN_SCALE for w, b in zip(white, black, strict=True))
        _PAWN_CACHE[key] = cached
    return cached


def _piece_terms(board: chess.Board, color: chess.Color) -> List[float]:
    """Bishop pair, rook file, king shield and mobility terms of one colour.

    Rooks on a file without any pawn count as open, rooks on a file without a friendly pawn
    but with an enemy pawn count as half-open. The king shield counts friendly pawns on the
    three files around the king, one or two ranks ahead of it, scaled by ``1 / 3``. Mobility
    of a piece is the number of squares it attacks that hold no friendly piece (not legal
    moves), summed per piece type and scaled by ``1 / 10``.

    :param board: The position.
    :param color: The colour.
    :return: Values of the terms after ``"passed"``, ``"doubled"``, ``"isolated"``, in order.
    """
    own = board.occupied_co[color]
    own_pawns = board.pawns & own
    bishop_pair = 1.0 if board.pieces_mask(chess.BISHOP, color).bit_count() >= 2 else 0.0

    rooks_open = rooks_half_open = 0
    for square in chess.scan_forward(board.rooks & own):
        file_mask = chess.BB_FILES[chess.square_file(square)]
        if not board.pawns & file_mask:
            rooks_open += 1
        elif not own_pawns & file_mask:
            rooks_half_open += 1

    shield = 0
    king = board.king(color)
    if king is not None:
        king_file, king_rank = chess.square_file(king), chess.square_rank(king)
        direction = 1 if color == chess.WHITE else -1
        for file in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
            for rank in (king_rank + direction, king_rank + 2 * direction):
                if 0 <= rank < 8 and (own_pawns >> chess.square(file, rank)) & 1:
                    shield += 1

    terms = [bishop_pair, float(rooks_open), float(rooks_half_open), shield / _SHIELD_SCALE]
    for piece_type in _MOBILITY_PIECES:
        mobility = 0
        for square in chess.scan_forward(board.pieces_mask(piece_type, color)):
            mobility += (board.attacks_mask(square) & ~own).bit_count()
        terms.append(mobility / _MOBILITY_SCALE)
    return terms


class PstTermsFeatures(LinearFeatures):
    """Tapered piece-square tables plus classical terms, one middlegame and one endgame weight each.

    The 768 piece-square entries are those of :class:`PstFeatures`. The terms of
    :data:`TERM_NAMES` follow: each is the White value minus the Black value (passed, doubled
    and isolated pawns, bishop pair, rooks on open and half-open files, king shield, and the
    mobility of each minor and major piece), oriented by perspective exactly as the piece
    values are. Each term adds ``value * phase`` to its middlegame weight and
    ``value * (1 - phase)`` to its endgame weight. Under ``"white"`` one turn entry (``+1``
    White to move, ``-1`` Black) follows the terms.

    :cvar name: ``"pst_terms"``.
    """

    name = "pst_terms"
    _TERM_BASE = 2 * BLOCK_SIZE
    _TURN_INDEX = _TERM_BASE + 2 * len(TERM_NAMES)
    _pst = PstFeatures()

    def size(self, perspective: str) -> int:
        """PST weights, two weights per term, plus one turn weight under ``"white"``.

        :param perspective: ``"white"`` or ``"side_to_move"``.
        :return: ``790`` or ``791``.
        """
        return self._TURN_INDEX + (1 if perspective == "white" else 0)

    def active(self, board: chess.Board, perspective: str) -> Tuple[List[int], List[float]]:
        """Entries of the piece-square tables, of every term, and the turn feature.

        :param board: The position.
        :param perspective: ``"white"`` or ``"side_to_move"``.
        :return: Two parallel lists; every term gives two entries, even when its value is zero.
        """
        indices, values = self._pst.active(board, perspective)
        if perspective == "white":
            # PstFeatures ends with its own turn entry; this class appends a later one.
            indices.pop()
            values.pop()
        tapered = phase(board)
        white = _piece_terms(board, chess.WHITE)
        black = _piece_terms(board, chess.BLACK)
        terms = list(_pawn_terms(board)) + [w - b for w, b in zip(white, black, strict=True)]
        sign = 1.0 if perspective == "white" or board.turn == chess.WHITE else -1.0
        for term, value in enumerate(terms):
            index = self._TERM_BASE + 2 * term
            indices.extend((index, index + 1))
            values.extend((sign * value * tapered, sign * value * (1.0 - tapered)))
        if perspective == "white":
            indices.append(self._TURN_INDEX)
            values.append(1.0 if board.turn == chess.WHITE else -1.0)
        return indices, values


#: Offset of the opponent's piece block in :class:`SlottedPstFeatures`.
_OPPONENT_PIECE_BLOCK = 6 * 64


class SlottedPstFeatures(LinearFeatures):
    """One-hot piece-square features with one entry per piece, for neural inputs.

    Under ``"side_to_move"`` a piece of the player to move gets index ``type * 64 + square`` and a
    piece of the opponent gets ``384 + type * 64 + square``, the square seen from the home rank of
    its colour (mirrored by rank for Black). Every value is ``1.0``, so the vector has at most
    :attr:`max_active` non-zero entries.

    Under ``"white"`` no mirroring is applied: White pieces use the first block, Black pieces the
    second, and one turn entry at index 768 follows: ``1.0`` when White is to move, ``0.0`` when
    Black is to move.

    :cvar name: ``"pst_slotted"``.
    :cvar max_active: Most entries under ``"side_to_move"`` (one per piece, 32 at most).
    """

    name = "pst_slotted"
    max_active = 32
    _TURN_INDEX = 2 * _OPPONENT_PIECE_BLOCK

    def max_entries(self, perspective: str) -> int:
        """Most non-zero entries a position can have under ``perspective``.

        :param perspective: ``"white"`` or ``"side_to_move"``.
        :return: ``32`` under ``"side_to_move"`` (one per piece), ``33`` under ``"white"`` (plus
            the turn entry).
        """
        return self.max_active + (1 if perspective == "white" else 0)

    def size(self, perspective: str) -> int:
        """Two blocks of 384 piece-square weights, plus one turn weight under ``"white"``.

        :param perspective: ``"white"`` or ``"side_to_move"``.
        :return: ``768`` or ``769``.
        """
        return self._TURN_INDEX + (1 if perspective == "white" else 0)

    def active(self, board: chess.Board, perspective: str) -> Tuple[List[int], List[float]]:
        """Index of every piece, all values ``1.0``, plus the turn entry under ``"white"``.

        :param board: The position.
        :param perspective: ``"white"`` or ``"side_to_move"``.
        :return: Two parallel lists; one entry per piece (plus the turn entry under ``"white"``).
        """
        mover = board.turn
        indices: List[int] = []
        for square, piece in board.piece_map().items():
            offset = (piece.piece_type - 1) * 64
            if perspective == "white":
                if piece.color == chess.BLACK:
                    offset += _OPPONENT_PIECE_BLOCK
                indices.append(offset + square)
                continue
            if piece.color != mover:
                offset += _OPPONENT_PIECE_BLOCK
            oriented = chess.square_mirror(square) if piece.color == chess.BLACK else square
            indices.append(offset + oriented)
        values = [1.0] * len(indices)
        if perspective == "white":
            indices.append(self._TURN_INDEX)
            values.append(1.0 if mover == chess.WHITE else 0.0)
        return indices, values


_KB_BUCKETS = 8
_KB_PIECE_CLASSES = 10
#: Features of one (slot, bucket) block: ten piece classes times 64 squares.
_KB_BLOCK = _KB_PIECE_CLASSES * 64
_KB_KING_BASE = 2 * _KB_BUCKETS * _KB_BLOCK
_KB_SIZE = _KB_KING_BASE + 2 * 64
_KB_WHITE_ERROR = (
    "king_bucket features are defined for the side_to_move perspective only, not for white"
)
#: Square maps indexed by ``2 * black + flip``: rank-oriented to the home rank (Black mirrored by
#: rank), then mirrored in file when ``flip`` is set (the king sits on files a to d).
_KB_SQUARE_MAPS = tuple(
    tuple((chess.square_mirror(sq) if black else sq) ^ (7 if flip else 0) for sq in range(64))
    for black in (0, 1)
    for flip in (0, 1)
)
#: Bucket of a king square, indexed by ``black`` then square: 2 file classes, home or advanced.
_KB_KING_BUCKET = tuple(
    tuple(
        2 * min(sq & 7, 7 - (sq & 7))
        + (1 if (chess.square_mirror(sq) if black else sq) >> 3 >= 2 else 0)
        for sq in range(64)
    )
    for black in (0, 1)
)


class KingBucketFeatures(LinearFeatures):
    """King-bucketed one-hot piece-square features, one set per side (HalfKP-like).

    Two slots are filled: slot 0 for the player to move and slot 1 for the opponent. Each is seen
    from its own home rank, and the king of the slot selects a bucket: ``2 * min(file, 7 - file)``
    plus one when the king is advanced (rank 3 or higher from its home rank). Every non-king piece
    adds one entry at ``(((slot * 8 + bucket) * 10 + piece_class) * 64 + square)``, where
    ``piece_class`` is the piece type (P, N, B, R, Q) for a friend of the slot and that type plus
    five for an enemy. The square is mirrored in file when the king of the slot stands on files
    e to h, so the king always lies on files a to d. Each slot adds one more entry for its king,
    at ``10240 + slot * 64 + square`` with the same file mirror. All values are ``1.0``.

    Only ``"side_to_move"`` is supported: the features are defined from the point of view of the
    player to move.

    :cvar name: ``"king_bucket"``.
    :cvar max_active: Most entries: two per non-king piece (at most 30 non-king pieces in total)
        plus two king entries, which is 62; 64 leaves room.
    """

    name = "king_bucket"
    max_active = 64

    def size(self, perspective: str) -> int:
        """The number of king-bucketed weights, under ``"side_to_move"`` only.

        :param perspective: ``"white"`` or ``"side_to_move"``.
        :return: ``10368``.
        :raises ValueError: If ``perspective`` is ``"white"``.
        """
        if perspective == "white":
            raise ValueError(_KB_WHITE_ERROR)
        return _KB_SIZE

    def active(self, board: chess.Board, perspective: str) -> Tuple[List[int], List[float]]:
        """Indices of the non-king pieces of both slots and of the two kings; all values ``1.0``.

        :param board: The position.
        :param perspective: ``"side_to_move"``.
        :return: Two parallel lists of length ``2 * (non-king pieces) + 2``.
        :raises ValueError: If ``perspective`` is ``"white"``.
        """
        if perspective == "white":
            raise ValueError(_KB_WHITE_ERROR)
        mover = board.turn
        indices: List[int] = []
        squares = []  # square map of each slot
        bases = []  # offset of the (slot, bucket) block of each slot
        for slot, color in enumerate((mover, not mover)):
            black = 0 if color == chess.WHITE else 1
            king = board.king(color)
            flip = 1 if (king & 7) >= 4 else 0
            square_map = _KB_SQUARE_MAPS[2 * black + flip]
            squares.append(square_map)
            bases.append((slot * _KB_BUCKETS + _KB_KING_BUCKET[black][king]) * _KB_BLOCK)
            indices.append(_KB_KING_BASE + slot * 64 + square_map[king])
        mover_map, opponent_map = squares
        mover_base, opponent_base = bases
        for square, piece in board.piece_map().items():
            if piece.piece_type == chess.KING:
                continue
            kind = piece.piece_type - 1
            if piece.color == mover:
                indices.append(mover_base + kind * 64 + mover_map[square])
                indices.append(opponent_base + (kind + 5) * 64 + opponent_map[square])
            else:
                indices.append(mover_base + (kind + 5) * 64 + mover_map[square])
                indices.append(opponent_base + kind * 64 + opponent_map[square])
        return indices, [1.0] * len(indices)
