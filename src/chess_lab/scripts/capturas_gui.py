"""Save PNG screenshots of the window, drawn offscreen with a fixed position.

The README shows these images; running this script again after a change to the window refreshes
them. No engine process is started: the AI is replaced by a stub, and the clocks are off so that
two runs give the same picture.

Example::

    QT_QPA_PLATFORM=offscreen python -m chess_lab.scripts.capturas_gui
    QT_QPA_PLATFORM=offscreen python -m chess_lab.scripts.capturas_gui --out /tmp/img
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import chess

from chess_engine.paths import ROOT

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from chess_engine.gui import game_window, theme  # noqa: E402
from chess_engine.gui.game_window import GamePage  # noqa: E402
from chess_engine.gui.settings import AISettings, MatchSetup, PlayerSetup  # noqa: E402
from chess_engine.gui.settings_dialog import AISettingsDialog  # noqa: E402
from chess_engine.gui.setup_dialog import SetupPage  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "img"
# Italian game, eight half-moves: White (the person) to move.
GAME_MOVES = "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6".split()
SIZES = {"setup": (520, 420), "game": (1080, 720), "settings": (460, 560)}


class NoEngine:
    """Stands in for ``UciBot``: the screenshots never need a move from the engine."""

    def __init__(self, settings):
        self.settings = settings

    def on_game_start(self) -> None:
        """Nothing to reset."""

    def close(self) -> None:
        """Nothing to close."""


def make_game() -> GamePage:
    """A game in progress: a person as White against the AI, after eight half-moves, no clock.

    :return: The game page, with the moves already on the board.
    """
    real_bot = game_window.UciBot
    game_window.UciBot = NoEngine
    try:
        page = GamePage(MatchSetup(white=PlayerSetup(), black=PlayerSetup(AISettings()), base=None))
    finally:
        game_window.UciBot = real_bot
    for uci in GAME_MOVES:
        page.model.make_move(chess.Move.from_uci(uci))
    page._refresh()
    return page


def make_views() -> Dict[str, QWidget]:
    """The three screens of the window.

    :return: Widgets keyed by the name of the PNG (``setup``, ``game``, ``settings``).
    """
    return {"setup": SetupPage(), "game": make_game(), "settings": AISettingsDialog(AISettings())}


def save_all(out: Path) -> List[Path]:
    """Draw every view and write ``gui-<name>.png`` under ``out``.

    :param out: Output directory (created if missing).
    :return: The files written.
    """
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(theme.STYLESHEET)
    out.mkdir(parents=True, exist_ok=True)
    views = make_views()
    written = []
    try:
        for name, widget in views.items():
            widget.resize(*SIZES[name])
            widget.show()
            app.processEvents()
            path = out / f"gui-{name}.png"
            if not widget.grab().save(str(path)):
                raise OSError(f"could not write {path}")
            written.append(path)
    finally:
        views["game"].close_game()
    return written


def main(argv: Optional[List[str]] = None) -> int:
    """Write the screenshots.

    :param argv: Arguments; None reads ``sys.argv``.
    :return: Exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    for path in save_all(parser.parse_args(argv).out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
