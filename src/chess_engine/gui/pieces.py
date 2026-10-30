"""The piece drawings (``assets/pieces``, cburnett set) as Qt renderers, pixmaps and icons."""

from functools import cache
from pathlib import Path

import chess
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

PIECES_DIR = Path(__file__).parent / "assets" / "pieces"


def piece_file(piece: chess.Piece) -> Path:
    """File of the drawing of a piece.

    :param piece: Piece to draw.
    :return: Path of its SVG, e.g. ``wK.svg``.
    """
    return PIECES_DIR / f"{'w' if piece.color else 'b'}{piece.symbol().upper()}.svg"


@cache
def renderer(piece: chess.Piece) -> QSvgRenderer:
    """SVG renderer of a piece, loaded once.

    :param piece: Piece to draw.
    :return: Its renderer.
    """
    return QSvgRenderer(str(piece_file(piece)))


def draw_piece(painter: QPainter, piece: chess.Piece, rect: QRectF) -> None:
    """Paint a piece into a rectangle.

    :param painter: Painter of the widget.
    :param piece: Piece to draw.
    :param rect: Where to draw it.
    """
    renderer(piece).render(painter, rect)


def piece_icon(piece: chess.Piece, size: int = 48) -> QIcon:
    """Icon of a piece, for menus and buttons.

    :param piece: Piece to draw.
    :param size: Side in pixels.
    :return: The icon.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    draw_piece(painter, piece, QRectF(0, 0, size, size))
    painter.end()
    return QIcon(pixmap)
