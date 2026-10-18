"""Data representation: how Stockfish evaluations become training targets.

Raw evaluations (``cp`` / ``mate``) are always from **White's point of view**; a row has either
``cp`` or ``mate`` (the other is NaN). A :class:`Representation` fixes the scale of the target
(win probability or clipped linear centipawns), whose point of view it uses and which positions
are kept.
"""

import zlib
from dataclasses import asdict, dataclass

import chess
import numpy as np
import pandas as pd

from chess_engine.models.base import CP_SCALE

#: Slope of the Lichess win-probability curve (``1 / (1 + exp(-k * cp))``).
LICHESS_K = 0.00368208
#: Centipawns at which the linear scale saturates.
CP_LINEAR_SCALE = CP_SCALE

_SCALES = ("winprob", "cp_linear")
_PERSPECTIVES = ("white", "side_to_move")
_FILTERS = ("all", "quiet")
_SCORE_EPS = 1e-6
#: Salt of the hash that picks the mates kept by :func:`keep_mate_rows`.
_MATE_SALT = "mate-keep"
#: Default share of the mate positions kept in training (the 100 % variant needs ``--tag``).
DEFAULT_MATE_KEEP = 0.10


def validate_mate_keep(mate_keep: float) -> None:
    """Check the share of mate positions kept.

    :param mate_keep: Share in ``(0, 1]``; 1 keeps every mate.
    :raises ValueError: If it is outside ``(0, 1]``.
    """
    if not 0.0 < mate_keep <= 1.0:
        raise ValueError(f"mate_keep must be in (0, 1], got {mate_keep}")


@dataclass(frozen=True)
class Representation:
    """Choice of target scale, point of view and position filter.

    :param scale: ``"winprob"`` (logistic of centipawns) or ``"cp_linear"`` (clipped linear).
    :param k: Slope of the logistic curve (only used by ``"winprob"``).
    :param perspective: ``"white"`` or ``"side_to_move"`` (score of the player to move).
    :param filter: ``"all"`` positions or only ``"quiet"`` ones (see :func:`is_quiet`).
    :param mate_keep: Share of the mate positions kept in the training data, in ``(0, 1]``
        (see :func:`keep_mate_rows`); 1 keeps them all. Defaults to :data:`DEFAULT_MATE_KEEP`.
    """

    scale: str = "winprob"
    k: float = LICHESS_K
    perspective: str = "side_to_move"
    filter: str = "all"
    mate_keep: float = DEFAULT_MATE_KEEP

    def __post_init__(self) -> None:
        if self.scale not in _SCALES:
            raise ValueError(f"unknown scale: {self.scale!r}")
        if self.perspective not in _PERSPECTIVES:
            raise ValueError(f"unknown perspective: {self.perspective!r}")
        if self.filter not in _FILTERS:
            raise ValueError(f"unknown filter: {self.filter!r}")
        validate_mate_keep(self.mate_keep)

    @property
    def name(self) -> str:
        """Short identifier, e.g. ``"winprob-k0.00368-stm-all"``; ``-mates5`` if mates are cut."""
        scale = f"winprob-k{self.k:.5f}" if self.scale == "winprob" else self.scale
        persp = "stm" if self.perspective == "side_to_move" else "white"
        mates = f"-mates{100 * self.mate_keep:g}" if self.mate_keep < 1.0 else ""
        return f"{scale}-{persp}-{self.filter}{mates}"

    def to_dict(self) -> dict:
        """Fields as a plain dictionary.

        :return: ``{"scale": ..., "k": ..., "perspective": ..., "filter": ..., "mate_keep": ...}``.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Representation":
        """Build a representation from a dictionary.

        :param d: Dictionary produced by :meth:`to_dict`.
        :return: The equivalent representation.
        """
        return cls(**d)


def white_score(cp: np.ndarray, mate: np.ndarray, rep: Representation) -> np.ndarray:
    """Expected score for White in ``[0, 1]``.

    :param cp: Centipawns, White's point of view (NaN on mate rows).
    :param mate: Moves to mate, positive when White mates (NaN on non-mate rows).
    :param rep: Representation that selects the scale.
    :return: Mates give ``1.0`` (White mates) or ``0.0``; the rest follow ``rep.scale``.
    """
    cp = np.asarray(cp, dtype=np.float64)
    mate = np.asarray(mate, dtype=np.float64)
    is_mate = ~np.isnan(mate)
    safe_cp = np.where(np.isnan(cp), 0.0, cp)
    if rep.scale == "winprob":
        score = 1.0 / (1.0 + np.exp(-rep.k * safe_cp))
    else:
        score = 0.5 + 0.5 * np.clip(safe_cp / CP_LINEAR_SCALE, -1.0, 1.0)
    return np.where(is_mate, (mate > 0).astype(np.float64), score)


def target(
    cp: np.ndarray, mate: np.ndarray, white_to_move: np.ndarray, rep: Representation
) -> np.ndarray:
    """Training target under ``rep``.

    :param cp: Centipawns, White's point of view.
    :param mate: Moves to mate, White's point of view.
    :param white_to_move: Boolean array, True where White is to move.
    :param rep: Representation.
    :return: :func:`white_score`, as ``1 - s`` where Black moves if the perspective is
        ``"side_to_move"``.
    """
    score = white_score(cp, mate, rep)
    if rep.perspective == "side_to_move":
        score = np.where(np.asarray(white_to_move, dtype=bool), score, 1.0 - score)
    return score


def score_to_cp(score: float | np.ndarray, rep: Representation) -> float | np.ndarray:
    """Inverse of :func:`white_score` for non-mate scores.

    :param score: Score(s) in ``[0, 1]`` (clipped to ``[1e-6, 1 - 1e-6]``).
    :param rep: Representation that selects the scale.
    :return: Centipawns.
    """
    s = np.clip(score, _SCORE_EPS, 1.0 - _SCORE_EPS)
    if rep.scale == "winprob":
        return np.log(s / (1.0 - s)) / rep.k
    return (2.0 * s - 1.0) * CP_LINEAR_SCALE


def is_quiet(board: chess.Board, best_move: str | None) -> bool:
    """Whether a position is quiet.

    :param board: The position.
    :param best_move: Best move in UCI notation, or None if unknown.
    :return: True if not in check and the best move is neither a capture nor a promotion.
    """
    if board.is_check():
        return False
    if best_move is None:
        return True
    move = chess.Move.from_uci(best_move)
    return not (board.is_capture(move) or move.promotion is not None)


def keep_mate_rows(fens: pd.Series, mate_keep: float, seed: int = 42) -> np.ndarray:
    """Which mate positions to keep when only a share ``mate_keep`` of them is wanted.

    Each FEN gets a CRC32 hash (salted with ``seed``) mapped to ``[0, 1)``; it is kept when the
    hash is below ``mate_keep``. The kept set of a smaller share is therefore a subset of a larger
    one. The mask covers every given row: the caller keeps the non-mate rows itself.

    :param fens: FENs of the mate positions (any iterable of strings).
    :param mate_keep: Share to keep, in ``(0, 1]``.
    :param seed: Salt of the hash.
    :return: Boolean array, one entry per FEN.
    :raises ValueError: If ``mate_keep`` is outside ``(0, 1]``.
    """
    validate_mate_keep(mate_keep)
    salt = f"{_MATE_SALT}:{seed}:"
    hashes = np.fromiter((zlib.crc32(f"{salt}{fen}".encode()) for fen in fens), dtype=np.float64)
    return hashes / 2.0**32 < mate_keep
