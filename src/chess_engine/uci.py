"""Client of the UCI engine (``rust/target/release/uci``): the AI players of the game page."""

import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence

import chess

from chess_engine.knowledge.endgame_tablebase import EndgameTablebase
from chess_engine.knowledge.opening_book import OpeningBook

#: The engine binary, built with ``cargo build --release --bin uci`` in ``rust/``.
ENGINE = Path(__file__).resolve().parents[2] / "rust" / "target" / "release" / "uci"
#: Seconds per move when there is no clock and no fixed time.
NO_CLOCK_MOVE_TIME = 3.0


def _ms(seconds: float) -> int:
    return max(0, round(seconds * 1000))


class UciEngine:
    """One engine process, driven in UCI; a game keeps its table between moves.

    :param techniques: Techniques of selective search to switch on.
    :param threads: Threads that search.
    """

    def __init__(self, techniques: Sequence[str], threads: int):
        if not ENGINE.exists():
            raise FileNotFoundError(
                f"{ENGINE} is missing: build it with `cargo build --release --bin uci` in rust/"
            )
        self._process = subprocess.Popen(
            [str(ENGINE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Techniques value {','.join(techniques)}")
        self._send("isready")
        self._wait_for("readyok")

    def new_game(self) -> None:
        """Forget what the previous game left in the table."""
        self._send("ucinewgame")

    def bestmove(self, board: chess.Board, limits: str) -> Optional[chess.Move]:
        """Search the position the board shows.

        :param board: Position; its move stack is the game so far.
        :param limits: The arguments of ``go`` (depth, movetime or clock).
        :return: The move, or None if the engine found none.
        """
        command = f"position fen {board.root().fen()}"
        if board.move_stack:
            command += " moves " + " ".join(move.uci() for move in board.move_stack)
        self._send(command)
        self._send(f"go {limits}")
        uci = self._wait_for("bestmove").split()[1]
        return None if uci == "0000" else chess.Move.from_uci(uci)

    def close(self) -> None:
        """Let the engine exit: closing its input makes it finish and quit."""
        self._process.stdin.close()
        self._process.wait()

    def _send(self, line: str) -> None:
        self._process.stdin.write(line + "\n")
        self._process.stdin.flush()

    def _wait_for(self, word: str) -> str:
        for line in self._process.stdout:
            if line.startswith(word):
                return line
        raise RuntimeError(f"the engine stopped before answering {word!r}")


class UciBot:
    """An AI player: the opening book and the tablebases first, then the engine.

    :param settings: How the AI thinks (an :class:`~chess_engine.gui.settings.AISettings`).
    """

    def __init__(self, settings):
        self.settings = settings
        self.engine = UciEngine(settings.techniques, settings.threads)
        self.book = OpeningBook.shipped() if settings.opening_book else None
        self.tablebase = EndgameTablebase.shipped() if settings.tablebase else None
        self.in_book = self.book is not None

    def on_game_start(self) -> None:
        """Start a game: the engine forgets the last one and the book is followed again."""
        self.engine.new_game()
        self.in_book = self.book is not None

    def get_move(
        self,
        board: chess.Board,
        clocks: Optional[Dict[chess.Color, float]],
        increment: float,
    ) -> Optional[chess.Move]:
        """The move for the position.

        :param board: Current position.
        :param clocks: Seconds left for each colour, or None without a clock.
        :param increment: Seconds added after each move.
        :return: The move, or None if the engine found none.
        """
        if self.in_book:
            book_move = self.book.lookup(board)
            if book_move is not None:
                return book_move
            self.in_book = False
        if self.tablebase is not None:
            move = self.tablebase.best_move(board)
            if move is not None:
                return move
        return self.engine.bestmove(board, self._limits(clocks, increment))

    def close(self) -> None:
        """Stop the engine process."""
        self.engine.close()

    def _limits(self, clocks: Optional[Dict[chess.Color, float]], increment: float) -> str:
        parts = []
        if self.settings.max_depth is not None:
            parts.append(f"depth {self.settings.max_depth}")
        if self.settings.move_time is not None:
            parts.append(f"movetime {_ms(self.settings.move_time)}")
        elif clocks is not None:
            white, black = clocks[chess.WHITE], clocks[chess.BLACK]
            parts.append(
                f"wtime {_ms(white)} btime {_ms(black)} winc {_ms(increment)} binc {_ms(increment)}"
            )
        else:
            parts.append(f"movetime {_ms(NO_CLOCK_MOVE_TIME)}")
        return " ".join(parts)
