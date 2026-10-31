"""Perspective detection for the public Lichess evaluation database.

The database (https://database.lichess.org, CC0) is a ``.jsonl.zst`` file with one position per
line, evaluated with Stockfish. Its documentation does not say whose point of view the numbers
are from, so :func:`detect_perspective` measures it from the data instead of assuming it.
"""

from collections.abc import Iterable
from typing import Any

import chess
import numpy as np

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
WHITE, SIDE_TO_MOVE = "white", "side-to-move"
#: Records inspected to detect the perspective.
DETECTION_SAMPLE = 5000


def material_difference(fen: str) -> int:
    """White material minus Black material, in pawns (kings excluded).

    :param fen: Position.
    :return: The balance, positive when White has more.
    """
    board = chess.Board(fen)
    return sum(
        PIECE_VALUES.get(piece.piece_type, 0) * (1 if piece.color == chess.WHITE else -1)
        for piece in board.piece_map().values()
    )


def detect_perspective(records: Iterable[dict[str, Any]]) -> str:
    """Find out whether ``cp`` is from White's or from the side to move's point of view.

    Looks at positions where Black is to move: if ``cp`` grows with White's material advantage
    the numbers are from White's point of view, if it shrinks they are from Black's (the side to
    move's).

    :param records: Records with at least ``fen`` and ``cp`` (``cp`` None for mates).
    :return: :data:`WHITE` or :data:`SIDE_TO_MOVE`.
    :raises ValueError: If there is not enough data to decide.
    """
    cps, balances = [], []
    for record in records:
        if record["cp"] is None or " b " not in f" {record['fen']} ":
            continue
        cps.append(record["cp"])
        balances.append(material_difference(record["fen"]))
    if len(cps) < 50 or np.std(balances) == 0 or np.std(cps) == 0:
        raise ValueError("not enough Black-to-move positions to detect the perspective")
    correlation = np.corrcoef(cps, balances)[0, 1]
    return WHITE if correlation > 0 else SIDE_TO_MOVE
