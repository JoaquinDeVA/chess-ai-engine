"""The game page: board, clocks, move list, game buttons and the bots that play."""

from typing import Optional

import chess
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from chess_engine.bots.chess_bot import ChessBot
from chess_engine.gui.board_widget import BoardPanel, BoardWidget, PlayerBar
from chess_engine.gui.game_model import Clock, GameModel, ai_accepts_draw
from chess_engine.gui.settings import MatchSetup, build_bot

#: Pause between moves when two AIs play, by the name shown in the speed menu.
SPEEDS = {"Fast": 0.0, "Normal": 0.8, "Slow": 2.0}


class MoveThread(QThread):
    """Asks a bot for a move without freezing the window."""

    found = Signal(int, object)  # token, chess.Move or None

    def __init__(self, token: int, bot: ChessBot, board: chess.Board, time_left: float):
        """Prepare the search.

        :param token: Number of the request; the page ignores answers to old ones.
        :param bot: Bot that moves.
        :param board: Position (a copy; the page keeps playing on its own).
        :param time_left: Seconds on the bot's clock (infinity without a clock).
        """
        super().__init__()
        self.token, self.bot, self.board, self.time_left = token, bot, board, time_left

    def run(self) -> None:
        """Search and report; an error is reported as no move."""
        try:
            move = self.bot.get_move(self.board, self.time_left)
        except Exception:  # the page shows a message; the thread must not die silently
            move = None
        self.found.emit(self.token, move)


class GamePage(QWidget):
    """One game between the players of a :class:`MatchSetup`.

    :ivar new_game: Emitted when the user wants to choose other players.
    :ivar rematch: Emitted when the user wants to play the same setup again.
    """

    new_game = Signal()
    rematch = Signal()

    def __init__(self, setup: MatchSetup, parent=None):
        """Build the page and start the game.

        :param setup: Players, clock and starting position.
        :param parent: Parent widget.
        """
        super().__init__(parent)
        self.setup = setup
        clock = Clock(setup.base, setup.increment) if setup.base is not None else None
        self.model = GameModel(setup.fen, clock)
        self.players = {chess.WHITE: setup.white, chess.BLACK: setup.black}
        self.bots = {c: build_bot(p.ai) for c, p in self.players.items() if p.ai is not None}
        self.names = {c: p.label for c, p in self.players.items()}
        if setup.white.label == setup.black.label:
            self.names = {
                chess.WHITE: f"{self.names[chess.WHITE]} (White)",
                chess.BLACK: f"{self.names[chess.BLACK]} (Black)",
            }
        for color, bot in self.bots.items():
            bot.on_game_start(color, self.names[not color])
        self._token = 0
        self._thread: Optional[MoveThread] = None
        self._view: Optional[int] = None  # ply being looked at; None = the live position
        self._ended = False
        self._build()
        human = [c for c, p in self.players.items() if p.is_human]
        self.board.set_orientation(human[0] if human else chess.WHITE)
        self._place_bars()
        self._refresh()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(100)
        self._next()

    # -- construction ---------------------------------------------------------------------

    def _build(self) -> None:
        with_clock = self.model.clock is not None
        self.board = BoardWidget(self.model.legal_moves_from)
        self.board.move_requested.connect(self._human_move)
        self.bars = {c: PlayerBar(c, self.names[c], with_clock) for c in (chess.WHITE, chess.BLACK)}
        self.panel = BoardPanel(self.board, self.bars[chess.BLACK], self.bars[chess.WHITE])

        self.moves = QTableWidget(0, 3)
        self.moves.setHorizontalHeaderLabels(["#", "White", "Black"])
        self.moves.verticalHeader().setVisible(False)
        self.moves.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.moves.setSelectionMode(QAbstractItemView.NoSelection)
        self.moves.setFocusPolicy(Qt.NoFocus)
        self.moves.horizontalHeader().setStretchLastSection(True)
        self.moves.setColumnWidth(0, 40)
        self.moves.cellClicked.connect(self._cell_clicked)

        self.banner = QLabel("")
        self.banner.setObjectName("banner")
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        self.status = QLabel("")
        self.status.setObjectName("muted")

        self.speed = QComboBox()
        self.speed.addItems(list(SPEEDS))
        self.speed.setCurrentText("Normal")
        self.speed.setVisible(not self.setup.has_humans())

        nav = QHBoxLayout()
        for text, ply in (("⏮", "start"), ("◀", "back"), ("▶", "forward"), ("⏭", "end")):
            button = QPushButton(text)
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(lambda _=False, where=ply: self._navigate(where))
            nav.addWidget(button)

        self.undo_button = self._button("Undo", self._undo)
        self.draw_button = self._button("Draw", self._offer_draw)
        self.resign_button = self._button("Resign", self._resign, "danger")
        self.flip_button = self._button("Flip", self._flip)
        self.pgn_button = self._button("Copy PGN", self._copy_pgn)
        has_human = self.setup.has_humans()
        for button in (self.undo_button, self.draw_button, self.resign_button):
            button.setVisible(has_human)

        self.again_button = self._button("Rematch", self.rematch.emit, "primary")
        self.new_button = self._button("New game", self.new_game.emit)
        self.again_button.setVisible(False)
        self.new_button.setVisible(True)

        side = QVBoxLayout()
        side.addWidget(self.banner)
        side.addWidget(self.moves, 1)
        side.addLayout(nav)
        side.addWidget(self.status)
        side.addWidget(self.speed)
        row = QHBoxLayout()
        row.addWidget(self.undo_button)
        row.addWidget(self.draw_button)
        row.addWidget(self.resign_button)
        side.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.flip_button)
        row.addWidget(self.pgn_button)
        side.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.again_button)
        row.addWidget(self.new_button)
        side.addLayout(row)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(300)

        layout = QHBoxLayout(self)
        layout.addWidget(self.panel, 1)
        layout.addWidget(side_widget)
        self.setFocusPolicy(Qt.StrongFocus)

    def _button(self, text: str, slot, name: str = "") -> QPushButton:
        button = QPushButton(text)
        button.setFocusPolicy(Qt.NoFocus)
        if name:
            button.setObjectName(name)
        button.clicked.connect(lambda _=False: slot())
        return button

    def _place_bars(self) -> None:
        bottom = self.board.orientation
        self.panel.set_bars(self.bars[not bottom], self.bars[bottom])

    # -- showing the position -------------------------------------------------------------

    def _position(self) -> chess.Board:
        """The board being looked at (the live one, or an earlier ply)."""
        live = self.model.board
        if self._view is None or self._view >= len(live.move_stack):
            return live
        board = live.copy(stack=True)
        while len(board.move_stack) > self._view:
            board.pop()
        return board

    def _refresh(self) -> None:
        board = self._position()
        last = board.peek() if board.move_stack else None
        self.board.set_board(board, last)
        self._fill_moves()
        self._update_controls()
        self._tick()

    def _fill_moves(self) -> None:
        sans = self.model.san_moves
        self.moves.setRowCount((len(sans) + 1) // 2)
        shown = len(self._position().move_stack)
        for ply, san in enumerate(sans):
            row, column = divmod(ply, 2)
            if column == 0:
                self.moves.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            item = QTableWidgetItem(san)
            if ply + 1 == shown:
                item.setBackground(Qt.darkGray)
            self.moves.setItem(row, 1 + column, item)
        if sans:
            self.moves.scrollToItem(self.moves.item((shown - 1) // 2 if shown else 0, 1))

    def _update_controls(self) -> None:
        live = self._view is None
        turn = self.model.turn
        over = self.model.outcome() is not None
        mine = self.players[turn].is_human and live and not over
        self.board.set_movable(turn if mine else None)
        busy = self._thread is not None
        humans = [c for c, p in self.players.items() if p.is_human]
        can_act = bool(humans) and not over
        self.undo_button.setEnabled(
            can_act
            and not busy
            and self.players[turn].is_human
            and bool(self.model.board.move_stack)
        )
        self.draw_button.setEnabled(can_act and self.players[turn].is_human)
        self.resign_button.setEnabled(can_act)
        if not over:
            who = self.names[turn]
            self.status.setText(f"{who} is thinking…" if busy else f"{who} to move")

    def _tick(self) -> None:
        clock = self.model.clock
        if clock is not None:
            running = self.model.turn if self.model.outcome() is None else None
            for color, bar in self.bars.items():
                bar.set_clock(clock.remaining(color), color == running)
        if not self._ended and self.model.outcome() is not None:
            self._end()

    # -- playing --------------------------------------------------------------------------

    def _human_move(self, move: chess.Move) -> None:
        if self._ended or self._view is not None:
            return
        self._play(move)

    def _play(self, move: chess.Move) -> None:
        self.model.make_move(move)
        self._view = None
        self._refresh()
        self._next()

    def _next(self) -> None:
        """Ask the bot to move if it is its turn."""
        if self._ended or self._thread is not None:
            return
        if self.model.outcome() is not None:
            self._end()
            return
        turn = self.model.turn
        if turn not in self.bots:
            return
        delay = 0.0 if self.setup.has_humans() else SPEEDS[self.speed.currentText()]
        if delay and self.model.board.move_stack:
            if self.model.clock is not None:
                self.model.clock.stop()
            QTimer.singleShot(int(delay * 1000), lambda t=self._token: self._start_search(t))
        else:
            self._start_search(self._token)

    def _start_search(self, token: int) -> None:
        if token != self._token or self._ended or self._thread is not None:
            return
        turn = self.model.turn
        if turn not in self.bots:
            return
        clock = self.model.clock
        if clock is not None:
            clock.start(turn)
        left = clock.remaining(turn) if clock is not None else float("inf")
        self._thread = MoveThread(self._token, self.bots[turn], self.model.board.copy(), left)
        self._thread.found.connect(self._bot_moved)
        self._thread.start()
        self._update_controls()

    def _bot_moved(self, token: int, move: Optional[chess.Move]) -> None:
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.wait()
        if token != self._token or self._ended:
            return
        if move is None or move not in self.model.board.legal_moves:
            QMessageBox.warning(self, "Engine error", "The engine did not return a legal move.")
            self._finish_with(lambda: self.model.resign(self.model.turn))
            return
        if self.model.outcome() is not None:  # flagged while thinking
            self._refresh()
            return
        self._play(move)

    # -- buttons --------------------------------------------------------------------------

    def _undo(self) -> None:
        if self._thread is not None or not self.players[self.model.turn].is_human:
            return
        self._token += 1
        self.model.undo(2 if len(self.model.board.move_stack) >= 2 else 1)
        self._view = None
        self._refresh()
        self._next()

    def _offer_draw(self) -> None:
        if self.model.can_claim_draw():
            self._finish_with(self.model.agree_draw)
            return
        opponents = [c for c in self.bots if c != self.model.turn]
        if not opponents:
            return
        if ai_accepts_draw(self.model.board, opponents[0]):
            self._finish_with(self.model.agree_draw)
        else:
            QMessageBox.information(self, "Draw", f"{self.names[opponents[0]]} declines the draw.")

    def _resign(self) -> None:
        humans = [c for c, p in self.players.items() if p.is_human]
        color = self.model.turn if self.model.turn in humans else humans[0]
        self._finish_with(lambda: self.model.resign(color))

    def _finish_with(self, action) -> None:
        self._token += 1
        action()
        self._refresh()

    def _flip(self) -> None:
        self.board.set_orientation(not self.board.orientation)
        self._place_bars()

    def _copy_pgn(self) -> None:
        text = self.model.pgn(self.names[chess.WHITE], self.names[chess.BLACK])
        QGuiApplication.clipboard().setText(text)
        self.status.setText("PGN copied")

    # -- navigation -----------------------------------------------------------------------

    def _navigate(self, where: str) -> None:
        total = len(self.model.board.move_stack)
        now = total if self._view is None else self._view
        target = {"start": 0, "back": now - 1, "forward": now + 1, "end": total}[where]
        target = max(0, min(total, target))
        self._view = None if target == total else target
        self._refresh()

    def _cell_clicked(self, row: int, column: int) -> None:
        if column == 0:
            return
        ply = row * 2 + column  # ply count after the clicked move
        if ply <= len(self.model.san_moves):
            self._view = None if ply == len(self.model.san_moves) else ply
            self._refresh()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Left/right/Home/End walk through the game."""
        keys = {
            Qt.Key_Left: "back",
            Qt.Key_Right: "forward",
            Qt.Key_Home: "start",
            Qt.Key_End: "end",
        }
        if event.key() in keys:
            self._navigate(keys[event.key()])
        else:
            super().keyPressEvent(event)

    # -- end ------------------------------------------------------------------------------

    def _end(self) -> None:
        self._ended = True
        self._token += 1
        outcome = self.model.outcome()
        winner = outcome.winner
        who = "Draw" if winner is None else f"{self.names[winner]} wins"
        self.banner.setText(f"{who} — {outcome.reason} ({outcome.score})")
        self.banner.setVisible(True)
        self.again_button.setVisible(True)
        self.status.setText("Game over")
        self._update_controls()
        self.board.set_movable(None)

    def close_game(self) -> None:
        """Stop the clocks and wait for a running search; call before leaving the page."""
        self._token += 1
        self._ended = True
        self._timer.stop()
        if self._thread is not None:
            self._thread.wait()
            self._thread = None
