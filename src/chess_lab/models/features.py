"""Sparse board features for the network inputs, seen from the side to move.

A position becomes a list of active feature indices and their values; the first layer of the
network adds ``weights[index] * value`` over those entries.
"""

from typing import Dict, List, Tuple

import chess

#: Phase of the start position (sum of the phase weights of all pieces).
MAX_PHASE = 24


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


#: Names of the classical terms of :class:`HybridPstFeatures`, in input order (two entries each).
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


def _term_entries(board: chess.Board, base: int) -> Tuple[List[int], List[float]]:
    """Middlegame and endgame entries of the eleven classical terms, from ``base`` on.

    Each term is the mover's value minus the opponent's; it adds ``value * phase`` to its
    middlegame entry and ``value * (1 - phase)`` to its endgame entry.

    :param board: The position.
    :param base: Index of the first term entry.
    :return: Two parallel lists of ``2 * len(TERM_NAMES)`` entries; zero values are kept.
    """
    tapered = phase(board)
    white = _piece_terms(board, chess.WHITE)
    black = _piece_terms(board, chess.BLACK)
    terms = list(_pawn_terms(board)) + [w - b for w, b in zip(white, black, strict=True)]
    sign = 1.0 if board.turn == chess.WHITE else -1.0
    indices: List[int] = []
    values: List[float] = []
    for term, value in enumerate(terms):
        index = base + 2 * term
        indices.extend((index, index + 1))
        values.extend((sign * value * tapered, sign * value * (1.0 - tapered)))
    return indices, values


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
_OPPONENT_PIECE_BLOCK = 6 * 64


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
        return 2 * _OPPONENT_PIECE_BLOCK

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
                offset += _OPPONENT_PIECE_BLOCK
            oriented = chess.square_mirror(square) if piece.color == chess.BLACK else square
            indices.append(offset + oriented)
        return indices, [1.0] * len(indices)


class HybridPstFeatures(SlottedPstFeatures):
    """One-hot piece squares, per king bucket, plus the eleven classical terms as dense inputs.

    The slotted entries of :class:`SlottedPstFeatures` are shifted by ``768 * bucket``, where
    ``bucket`` depends on the square of the mover's king (mirrored by rank when it is Black): the
    file group (``8 / file_groups`` files wide) and whether the king stands on rank 3 or beyond.
    The terms of :data:`TERM_NAMES` follow, two entries each. With one bucket the king is ignored.

    :cvar name: ``"pst_hybrid"``.
    :cvar file_groups: King file groups; the buckets are ``2 * file_groups``, or ``1`` when ``0``.
    :cvar terms: Whether the term entries follow the piece squares.
    """

    name = "pst_hybrid"
    file_groups = 0
    terms = True

    @property
    def buckets(self) -> int:
        """Number of king buckets.

        :return: ``2 * file_groups``, or ``1`` without king buckets.
        """
        return 2 * self.file_groups if self.file_groups else 1

    @property
    def _term_base(self) -> int:
        """Index of the first term entry (after every bucket of piece squares)."""
        return self.buckets * 2 * _OPPONENT_PIECE_BLOCK

    def bucket(self, king: int) -> int:
        """Bucket of an own king.

        :param king: Square of the king, as seen from the home rank of its colour.
        :return: Bucket in ``[0, buckets)``.
        """
        if not self.file_groups:
            return 0
        group = chess.square_file(king) // (8 // self.file_groups)
        return 2 * group + (1 if chess.square_rank(king) >= 2 else 0)

    @property
    def _term_count(self) -> int:
        """Number of dense entries: two per classical term, or none without terms."""
        return 2 * len(TERM_NAMES) if self.terms else 0

    def _dense_entries(self, board: chess.Board) -> Tuple[List[int], List[float]]:
        """The dense entries that follow the piece squares (the classical terms, if any)."""
        if not self.terms:
            return [], []
        return _term_entries(board, self._term_base)

    def max_entries(self) -> int:
        """Pieces and the term entries.

        :return: ``32 + 22``, or ``32`` without terms.
        """
        return self.max_active + self._term_count

    def size(self) -> int:
        """Bucketed piece squares and two entries per term.

        :return: ``768 * buckets + 22``, or ``768 * buckets`` without terms.
        """
        return self._term_base + self._term_count

    def active(self, board: chess.Board) -> Tuple[List[int], List[float]]:
        """Shifted piece entries, then the term entries.

        :param board: The position.
        :return: Two parallel lists.
        """
        indices, values = super().active(board)
        king = board.king(board.turn)
        if board.turn == chess.BLACK:
            king = chess.square_mirror(king)
        shift = self.bucket(king) * 2 * _OPPONENT_PIECE_BLOCK
        indices = [index + shift for index in indices]
        dense_indices, dense_values = self._dense_entries(board)
        indices.extend(dense_indices)
        values.extend(dense_values)
        return indices, values


class HybridKings4PstFeatures(HybridPstFeatures):
    """:class:`HybridPstFeatures` with 4 king buckets (2 file groups by 2 rank groups).

    :cvar name: ``"pst_hybrid_k4"``.
    """

    name = "pst_hybrid_k4"
    file_groups = 2


class HybridKings8PstFeatures(HybridPstFeatures):
    """:class:`HybridPstFeatures` with 8 king buckets (4 file groups by 2 rank groups).

    :cvar name: ``"pst_hybrid_k8"``.
    """

    name = "pst_hybrid_k8"
    file_groups = 4


class Kings4PstFeatures(HybridPstFeatures):
    """Piece squares in 4 king buckets (2 file groups by 2 rank groups), without the terms.

    :cvar name: ``"pst_k4"``.
    """

    name = "pst_k4"
    file_groups = 2
    terms = False


class ThreatsPstFeatures(HybridPstFeatures):
    """Piece squares per king bucket plus five threat inputs instead of the classical terms.

    The threats are the best capture of each side, the pieces each side has en prise and the
    check flag (:func:`_threat_entries`); they stand for what a search of captures would see.

    :cvar name: ``"pst_threats"``, with no king buckets.
    """

    name = "pst_threats"
    terms = False

    @property
    def _term_count(self) -> int:
        """Number of threat entries."""
        return THREAT_COUNT

    def _dense_entries(self, board: chess.Board) -> Tuple[List[int], List[float]]:
        """The threat entries after the piece squares."""
        return _threat_entries(board, self._term_base)


class Threats4PstFeatures(ThreatsPstFeatures):
    """:class:`ThreatsPstFeatures` with 4 king buckets (2 file groups by 2 rank groups).

    :cvar name: ``"pst_k4_threats"``.
    """

    name = "pst_k4_threats"
    file_groups = 2


#: Every feature set by its name, the one the saved models and the core refer to.
FEATURE_SETS: Dict[str, SlottedPstFeatures] = {
    features.name: features
    for features in (
        SlottedPstFeatures(),
        HybridPstFeatures(),
        HybridKings4PstFeatures(),
        HybridKings8PstFeatures(),
        Kings4PstFeatures(),
        ThreatsPstFeatures(),
        Threats4PstFeatures(),
    )
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
