"""The window: setup page and game page, and the entry point of the GUI."""

import sys
from typing import Optional

from PySide6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QStackedWidget, QWidget

from chess_engine.gui import theme
from chess_engine.gui.game_window import GamePage
from chess_engine.gui.settings import MatchSetup
from chess_engine.gui.setup_dialog import SetupPage


class MainWindow(QMainWindow):
    """Shows the setup page, and a game page while a game is on."""

    def __init__(self):
        """Create the window on the setup page."""
        super().__init__()
        self.setWindowTitle("ChessAIEngine")
        self.resize(1080, 720)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.setup_page = SetupPage()
        self.setup_page.start.connect(self.start_game)
        centred = QWidget()
        row = QHBoxLayout(centred)
        row.addStretch(1)
        row.addWidget(self.setup_page)
        row.addStretch(1)
        self.stack.addWidget(centred)
        self.centred = centred
        self.game: Optional[GamePage] = None

    def start_game(self, setup: MatchSetup) -> None:
        """Replace any game with a new one.

        :param setup: Players, clock and position of the game.
        """
        self._drop_game()
        self.game = GamePage(setup)
        self.game.new_game.connect(self.show_setup)
        self.game.rematch.connect(lambda: self.start_game(setup))
        self.stack.addWidget(self.game)
        self.stack.setCurrentWidget(self.game)

    def show_setup(self) -> None:
        """Go back to the player selection."""
        self._drop_game()
        self.stack.setCurrentWidget(self.centred)

    def _drop_game(self) -> None:
        if self.game is not None:
            self.game.close_game()
            self.stack.removeWidget(self.game)
            self.game.deleteLater()
            self.game = None

    def closeEvent(self, event) -> None:  # noqa: N802
        """Wait for a running search before closing."""
        self._drop_game()
        super().closeEvent(event)


def main() -> int:
    """Open the window and run until it is closed.

    :return: Exit code.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(theme.STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()
