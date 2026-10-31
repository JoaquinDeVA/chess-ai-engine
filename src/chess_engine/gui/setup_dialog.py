"""First page: who plays each colour, the time control and the AI settings."""

from typing import Dict

import chess
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from chess_engine.gui.pieces import piece_icon
from chess_engine.gui.settings import AISettings, MatchSetup, PlayerSetup, evaluation_label
from chess_engine.gui.settings_dialog import edit_settings

HUMAN, AI = "Human", "AI"

#: Time controls: label → (base seconds, increment seconds); None base = no clock.
TIME_CONTROLS = {
    "No clock": (None, 0.0),
    "1 min": (60.0, 0.0),
    "3 min + 2 s": (180.0, 2.0),
    "5 min": (300.0, 0.0),
    "10 min + 5 s": (600.0, 5.0),
    "15 min + 10 s": (900.0, 10.0),
    "30 min": (1800.0, 0.0),
}


class SetupPage(QWidget):
    """Choose the players and press Play.

    :ivar start: Emitted with the :class:`MatchSetup` when the user presses Play.
    """

    start = Signal(object)

    def __init__(self, parent=None):
        """Build the page: White is a person, Black an AI, ten minutes plus five seconds."""
        super().__init__(parent)
        self._ai: Dict[chess.Color, AISettings] = {
            chess.WHITE: AISettings(),
            chess.BLACK: AISettings(),
        }
        self._kind: Dict[chess.Color, QComboBox] = {}
        self._summary: Dict[chess.Color, QLabel] = {}
        self._button: Dict[chess.Color, QPushButton] = {}

        title = QLabel("ChessAIEngine")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        grid = QGridLayout()
        for row, color in ((0, chess.WHITE), (2, chess.BLACK)):
            icon = QLabel()
            icon.setPixmap(piece_icon(chess.Piece(chess.KING, color), 48).pixmap(48, 48))
            kind = QComboBox()
            kind.addItems([HUMAN, AI])
            kind.setCurrentText(HUMAN if color == chess.WHITE else AI)
            kind.currentTextChanged.connect(self._update)
            summary = QLabel("")
            summary.setObjectName("muted")
            button = QPushButton("⚙ Settings")
            button.clicked.connect(lambda _=False, c=color: self._edit(c))
            self._kind[color], self._summary[color], self._button[color] = kind, summary, button
            grid.addWidget(icon, row, 0)
            grid.addWidget(QLabel("White" if color else "Black"), row, 1)
            grid.addWidget(kind, row, 2)
            grid.addWidget(button, row, 3)
            grid.addWidget(summary, row + 1, 1, 1, 3)
        grid.setColumnStretch(2, 1)

        self.time_control = QComboBox()
        self.time_control.addItems(list(TIME_CONTROLS))
        self.time_control.setCurrentText("10 min + 5 s")
        swap = QPushButton("Swap sides")
        swap.clicked.connect(self._swap)
        self.hint = QLabel("")
        self.hint.setObjectName("muted")
        self.play = QPushButton("Play")
        self.play.setObjectName("primary")
        self.play.clicked.connect(lambda _=False: self.start.emit(self.match_setup()))

        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addLayout(grid)
        layout.addSpacing(10)
        layout.addWidget(QLabel("Time control"))
        layout.addWidget(self.time_control)
        layout.addWidget(swap)
        layout.addWidget(self.hint)
        layout.addWidget(self.play)
        layout.addStretch(2)
        self.setFixedWidth(640)
        self._update()

    def _swap(self) -> None:
        white, black = self._kind[chess.WHITE].currentText(), self._kind[chess.BLACK].currentText()
        self._kind[chess.WHITE].setCurrentText(black)
        self._kind[chess.BLACK].setCurrentText(white)
        self._ai[chess.WHITE], self._ai[chess.BLACK] = self._ai[chess.BLACK], self._ai[chess.WHITE]
        self._update()

    def _edit(self, color: chess.Color) -> None:
        name = "White" if color else "Black"
        chosen = edit_settings(self._ai[color], f"{name} AI settings", self)
        if chosen is not None:
            self._ai[color] = chosen
            self._update()

    def _update(self, *_) -> None:
        for color, kind in self._kind.items():
            is_ai = kind.currentText() == AI
            self._button[color].setEnabled(is_ai)
            settings = self._ai[color]
            time = (
                "share of the clock" if settings.move_time is None else f"{settings.move_time:g} s"
            )
            depth = (
                "no depth limit" if settings.max_depth is None else f"depth ≤ {settings.max_depth}"
            )
            self._summary[color].setText(
                f"{evaluation_label(settings.evaluation)} · {depth} · {time}"
                if is_ai
                else "You play with the mouse"
            )
        both_human = all(k.currentText() == HUMAN for k in self._kind.values())
        self.play.setEnabled(not both_human)
        self.hint.setText(
            "Human vs human is not supported: pick an AI for one side." if both_human else ""
        )

    def match_setup(self) -> MatchSetup:
        """The match the controls describe."""

        def player(color: chess.Color) -> PlayerSetup:
            return PlayerSetup(self._ai[color] if self._kind[color].currentText() == AI else None)

        base, increment = TIME_CONTROLS[self.time_control.currentText()]
        return MatchSetup(
            white=player(chess.WHITE), black=player(chess.BLACK), base=base, increment=increment
        )
