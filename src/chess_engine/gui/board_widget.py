"""The board of the window: squares, pieces, drag and drop, highlights and the player bars."""

from typing import Callable, List, Optional

import chess
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenu, QWidget

from chess_engine.gui import theme
from chess_engine.gui.pieces import draw_piece, piece_icon

PROMOTION_PIECES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)


class BoardWidget(QWidget):
    """A square board that shows a position and lets the user move one colour's pieces.

    The user picks a piece by clicking or dragging it; the legal targets show as dots and rings.
    A move is not played here: it is announced with :attr:`move_requested` and the owner shows the
    new position with :meth:`set_board`.
    """

    move_requested = Signal(object)  # chess.Move

    def __init__(self, legal_from: Callable[[chess.Square], List[chess.Move]], parent=None):
        """Create the board.

        :param legal_from: Legal moves of the piece on a square, in the position it shows.
        :param parent: Parent widget.
        """
        super().__init__(parent)
        self._legal_from = legal_from
        self._board = chess.Board()
        self._last_move: Optional[chess.Move] = None
        self._orientation = chess.WHITE
        self._movable: Optional[chess.Color] = None
        self._selected: Optional[chess.Square] = None
        self._targets: List[chess.Move] = []
        self._drag_from: Optional[chess.Square] = None
        self._drag_pos = QPointF()
        self.setMouseTracking(True)
        self.setMinimumSize(320, 320)

    def set_board(self, board: chess.Board, last_move: Optional[chess.Move]) -> None:
        """Show a position.

        :param board: Position to draw.
        :param last_move: Move to highlight (None for none).
        """
        self._board = board
        self._last_move = last_move
        self._clear_selection()
        self.update()

    def set_movable(self, color: Optional[chess.Color]) -> None:
        """Choose whose pieces the user may move now.

        :param color: Colour that may move, or None for nobody.
        """
        if color != self._movable:
            self._movable = color
            self._clear_selection()
            self.update()

    def set_orientation(self, color: chess.Color) -> None:
        """Put a colour at the bottom.

        :param color: Colour shown at the bottom.
        """
        self._orientation = color
        self.update()

    @property
    def orientation(self) -> chess.Color:
        """Colour at the bottom."""
        return self._orientation

    def _clear_selection(self) -> None:
        self._selected = None
        self._targets = []
        self._drag_from = None

    def _side(self) -> float:
        return float(min(self.width(), self.height()))

    def _rect(self, square: chess.Square) -> QRectF:
        size = self._side() / 8
        file, rank = chess.square_file(square), chess.square_rank(square)
        column = file if self._orientation == chess.WHITE else 7 - file
        row = 7 - rank if self._orientation == chess.WHITE else rank
        return QRectF(column * size, row * size, size, size)

    def _square_at(self, pos: QPointF) -> Optional[chess.Square]:
        size = self._side() / 8
        column, row = int(pos.x() // size), int(pos.y() // size)
        if not (0 <= column < 8 and 0 <= row < 8):
            return None
        if self._orientation == chess.WHITE:
            return chess.square(column, 7 - row)
        return chess.square(7 - column, row)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt name)
        """Draw squares, highlights, coordinates, pieces and the dragged piece."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        size = self._side() / 8
        light, dark = QColor(theme.BOARD_LIGHT), QColor(theme.BOARD_DARK)
        highlighted = {self._selected} if self._selected is not None else set()
        if self._last_move is not None:
            highlighted |= {self._last_move.from_square, self._last_move.to_square}

        font = QFont()
        font.setBold(True)
        font.setPixelSize(max(9, int(size * 0.17)))
        painter.setFont(font)
        for square in chess.SQUARES:
            rect = self._rect(square)
            is_light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
            painter.fillRect(rect, light if is_light else dark)
            if square in highlighted:
                painter.fillRect(rect, QColor(*theme.HIGHLIGHT))
            painter.setPen(dark if is_light else light)
            if rect.left() == 0:
                painter.drawText(
                    rect.adjusted(size * 0.05, size * 0.03, 0, 0),
                    Qt.AlignLeft | Qt.AlignTop,
                    str(chess.square_rank(square) + 1),
                )
            if rect.bottom() >= self._side() - 1:
                painter.drawText(
                    rect.adjusted(0, 0, -size * 0.05, -size * 0.03),
                    Qt.AlignRight | Qt.AlignBottom,
                    chess.FILE_NAMES[chess.square_file(square)],
                )

        self._paint_check(painter, size)
        for square, piece in self._board.piece_map().items():
            if square == self._drag_from:
                painter.setOpacity(0.35)
            draw_piece(painter, piece, self._rect(square))
            painter.setOpacity(1.0)
        self._paint_targets(painter, size)
        if self._drag_from is not None:
            piece = self._board.piece_at(self._drag_from)
            if piece is not None:
                big = size * 1.1
                draw_piece(
                    painter,
                    piece,
                    QRectF(self._drag_pos.x() - big / 2, self._drag_pos.y() - big / 2, big, big),
                )

    def _paint_check(self, painter: QPainter, size: float) -> None:
        if not self._board.is_check():
            return
        king = self._board.king(self._board.turn)
        if king is None:
            return
        rect = self._rect(king)
        gradient = QRadialGradient(rect.center(), size * 0.6)
        gradient.setColorAt(0.0, QColor(230, 30, 30, 235))
        gradient.setColorAt(0.65, QColor(230, 30, 30, 110))
        gradient.setColorAt(1.0, QColor(230, 30, 30, 0))
        painter.fillRect(rect, gradient)

    def _paint_targets(self, painter: QPainter, size: float) -> None:
        painter.setPen(Qt.NoPen)
        shade = QColor(0, 0, 0, 38)
        for square in {move.to_square for move in self._targets}:
            rect = self._rect(square)
            capture = self._board.piece_at(square) is not None or self._board.is_en_passant(
                next(m for m in self._targets if m.to_square == square)
            )
            if capture:
                path = QPainterPath()
                path.addEllipse(rect.adjusted(-0.0, -0.0, 0.0, 0.0))
                path.addEllipse(rect.adjusted(size * 0.1, size * 0.1, -size * 0.1, -size * 0.1))
                clip = QPainterPath()
                clip.addRect(rect)
                painter.setBrush(shade)
                painter.drawPath(path.intersected(clip))
            else:
                painter.setBrush(shade)
                radius = size * 0.16
                painter.drawEllipse(rect.center(), radius, radius)
        if self._drag_from is not None:
            hovered = self._square_at(self._drag_pos)
            if hovered is not None and hovered in {m.to_square for m in self._targets}:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(255, 255, 255, 190), max(2.0, size * 0.04)))
                painter.drawRect(self._rect(hovered).adjusted(1, 1, -1, -1))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Select a piece, or move the selected one onto the clicked square."""
        if event.button() == Qt.RightButton:
            self._clear_selection()
            self.update()
            return
        if event.button() != Qt.LeftButton:
            return
        square = self._square_at(event.position())
        if square is None:
            return
        if self._selected is not None and any(m.to_square == square for m in self._targets):
            self._try_move(square)
            return
        piece = self._board.piece_at(square)
        if piece is not None and piece.color == self._movable:
            self._selected = square
            self._targets = self._legal_from(square)
            self._drag_from = square
            self._drag_pos = event.position()
        else:
            self._clear_selection()
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        """Carry the dragged piece."""
        if self._drag_from is not None:
            self._drag_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """Drop the dragged piece: play the move if the square is a legal target."""
        if self._drag_from is None:
            return
        origin = self._drag_from
        self._drag_from = None
        target = self._square_at(event.position())
        if (
            target is not None
            and target != origin
            and any(m.to_square == target for m in self._targets)
        ):
            self._try_move(target)
        else:
            self.update()

    def _try_move(self, target: chess.Square) -> None:
        moves = [m for m in self._targets if m.to_square == target]
        move = moves[0]
        if len(moves) > 1:
            piece_type = self._ask_promotion(target)
            if piece_type is None:
                self._drag_from = None
                self.update()
                return
            move = next(m for m in moves if m.promotion == piece_type)
        self._clear_selection()
        self.update()
        self.move_requested.emit(move)

    def _ask_promotion(self, target: chess.Square) -> Optional[int]:
        menu = QMenu(self)
        menu.setStyleSheet("QMenu::icon { padding: 4px; }")
        for piece_type in PROMOTION_PIECES:
            action = menu.addAction(
                piece_icon(chess.Piece(piece_type, self._movable), 56), chess.piece_name(piece_type)
            )
            action.setData(piece_type)
        chosen = menu.exec(self.mapToGlobal(self._rect(target).center().toPoint()))
        return None if chosen is None else chosen.data()


class PlayerBar(QFrame):
    """Name, small king and clock of one player."""

    def __init__(self, color: chess.Color, name: str, with_clock: bool, parent=None):
        """Create the bar.

        :param color: Colour of the player (picks the king shown).
        :param name: Name to show.
        :param with_clock: Whether to show a clock.
        :param parent: Parent widget.
        """
        super().__init__(parent)
        self.setObjectName("playerbar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        icon = QLabel()
        icon.setPixmap(piece_icon(chess.Piece(chess.KING, color), 34).pixmap(34, 34))
        layout.addWidget(icon)
        label = QLabel(name)
        label.setObjectName("playername")
        layout.addWidget(label)
        layout.addStretch(1)
        self.clock = QLabel("")
        self.clock.setObjectName("clock")
        self.clock.setVisible(with_clock)
        layout.addWidget(self.clock)

    def set_clock(self, seconds: float, active: bool) -> None:
        """Show the time left.

        :param seconds: Seconds left.
        :param active: Whether this clock is running.
        """
        self.clock.setText(format_clock(seconds))
        self.clock.setProperty("active", active)
        self.clock.setProperty("low", seconds < 20)
        self.clock.style().unpolish(self.clock)
        self.clock.style().polish(self.clock)


def format_clock(seconds: float) -> str:
    """Time as ``m:ss``, ``h:mm:ss``, or with tenths under 20 seconds.

    :param seconds: Seconds left.
    :return: Text for the clock.
    """
    seconds = max(0.0, seconds)
    if seconds < 20:
        return f"0:{seconds:04.1f}"
    whole = int(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class BoardPanel(QWidget):
    """Top bar, board and bottom bar, with the board as large as fits and centred."""

    BAR_HEIGHT = 46

    def __init__(self, board: BoardWidget, top: PlayerBar, bottom: PlayerBar, parent=None):
        """Place the three widgets.

        :param board: The board.
        :param top: Bar above the board.
        :param bottom: Bar below the board.
        :param parent: Parent widget.
        """
        super().__init__(parent)
        self.board, self.top, self.bottom = board, top, bottom
        for child in (board, top, bottom):
            child.setParent(self)
        self.setMinimumSize(340, 340 + 2 * self.BAR_HEIGHT)

    def set_bars(self, top: PlayerBar, bottom: PlayerBar) -> None:
        """Swap which bar is above and which below (when the board flips).

        :param top: Bar to show above.
        :param bottom: Bar to show below.
        """
        self.top, self.bottom = top, bottom
        self._layout_children()

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Re-place the children."""
        self._layout_children()

    def _layout_children(self) -> None:
        side = max(0, min(self.width(), self.height() - 2 * self.BAR_HEIGHT))
        left = (self.width() - side) // 2
        total = side + 2 * self.BAR_HEIGHT
        y = (self.height() - total) // 2
        self.top.setGeometry(left, y, side, self.BAR_HEIGHT)
        self.board.setGeometry(left, y + self.BAR_HEIGHT, side, side)
        self.bottom.setGeometry(left, y + self.BAR_HEIGHT + side, side, self.BAR_HEIGHT)
