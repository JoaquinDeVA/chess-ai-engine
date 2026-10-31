"""Widgets of the GUI, drawn offscreen (skipped when PySide6 is not installed)."""

import os

import chess
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from chess_engine.gui.board_widget import BoardWidget, format_clock  # noqa: E402
from chess_engine.gui.game_window import GamePage  # noqa: E402
from chess_engine.gui.settings import AISettings, MatchSetup, PlayerSetup  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def click(widget: BoardWidget, square: chess.Square) -> None:
    rect = widget._rect(square)
    QTest.mouseClick(widget, Qt.LeftButton, pos=rect.center().toPoint())


def test_format_clock():
    assert format_clock(125) == "2:05"
    assert format_clock(3725) == "1:02:05"
    assert format_clock(7.34) == "0:07.3"
    assert format_clock(-1) == "0:00.0"


def test_board_click_to_move(app):
    model_board = chess.Board()
    board = BoardWidget(lambda sq: [m for m in model_board.legal_moves if m.from_square == sq])
    board.resize(400, 400)
    board.set_board(model_board, None)
    board.set_movable(chess.WHITE)
    got = []
    board.move_requested.connect(got.append)
    click(board, chess.E2)
    click(board, chess.E4)
    assert got == [chess.Move.from_uci("e2e4")]


def test_board_ignores_other_colour_and_orientation(app):
    model_board = chess.Board()
    board = BoardWidget(lambda sq: [m for m in model_board.legal_moves if m.from_square == sq])
    board.resize(400, 400)
    board.set_board(model_board, None)
    board.set_movable(chess.WHITE)
    got = []
    board.move_requested.connect(got.append)
    click(board, chess.E7)
    click(board, chess.E5)
    assert got == []
    board.set_orientation(chess.BLACK)
    assert board._square_at(QPointF(5, 5)) == chess.H1
    board.grab()  # paints without raising


def test_game_page_human_move_and_undo(app):
    setup = MatchSetup(
        white=PlayerSetup(),
        black=PlayerSetup(AISettings(max_depth=1, move_time=0.2)),
        base=None,
    )
    page = GamePage(setup)
    try:
        page.resize(900, 600)
        page._human_move(chess.Move.from_uci("e2e4"))
        QTest.qWait(100)
        for _ in range(100):
            if len(page.model.board.move_stack) == 2:
                break
            QTest.qWait(50)
        assert len(page.model.board.move_stack) == 2
        page._undo()
        assert len(page.model.board.move_stack) == 0
        page._navigate("end")
    finally:
        page.close_game()


def test_game_page_resign(app):
    setup = MatchSetup(
        white=PlayerSetup(),
        black=PlayerSetup(AISettings(max_depth=1, move_time=0.1)),
        base=60.0,
    )
    page = GamePage(setup)
    try:
        page._resign()
        assert page.model.outcome().score == "0-1"
        assert page.banner.isVisibleTo(page)
    finally:
        page.close_game()


def test_setup_page_builds_the_match(app):
    from chess_engine.gui.setup_dialog import AI, HUMAN, SetupPage

    page = SetupPage()
    setup = page.match_setup()
    assert setup.white.is_human and not setup.black.is_human
    assert (setup.base, setup.increment) == (600.0, 5.0)
    page._kind[chess.WHITE].setCurrentText(AI)
    page._kind[chess.BLACK].setCurrentText(HUMAN)
    page.time_control.setCurrentText("No clock")
    setup = page.match_setup()
    assert not setup.white.is_human and setup.black.is_human and setup.base is None
    page._kind[chess.WHITE].setCurrentText(HUMAN)
    assert not page.play.isEnabled()


def test_settings_dialog_round_trip(app):
    from chess_engine.gui.settings_dialog import AISettingsDialog

    original = AISettings(max_depth=5, move_time=None, tablebase=False)
    dialog = AISettingsDialog(original)
    assert dialog.settings() == original
    dialog.depth.setValue(7)
    dialog.techniques["lmr"].setChecked(True)
    changed = dialog.settings()
    assert changed.max_depth == 7 and "lmr" in changed.techniques


def test_main_window_starts_and_drops_a_game(app):
    from chess_engine.gui.app import MainWindow

    window = MainWindow()
    window.start_game(
        MatchSetup(
            white=PlayerSetup(),
            black=PlayerSetup(AISettings(max_depth=1, move_time=0.1)),
            base=None,
        )
    )
    assert window.stack.currentWidget() is window.game
    window.show_setup()
    assert window.game is None and window.stack.currentWidget() is window.centred
