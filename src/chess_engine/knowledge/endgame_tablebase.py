"""Endgame tablebases: perfect play in positions with few pieces.

A Syzygy tablebase stores, for every position with up to five pieces (kings included), whether the
side to move wins, draws or loses (WDL) and how many moves are left before the next capture or pawn
move (DTZ). With the 3-4-5 piece tables (about 940 MB) the engine never needs to search such a
position: :meth:`EndgameTablebase.best_move` returns a move that keeps the best result and, among
the winning moves, the one that finishes the quickest.

The tables are not versioned (they are large); see ``docs/DATA_AND_TRAINING.md`` for where they go.
They are probed by the Rust core (``rust/src/knowledge/tablebase.rs``); this class is its Python
face.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import chess

from chess_engine import _core

#: Default location of the tables.
DEFAULT_TABLEBASE_DIR = Path(__file__).resolve().parents[3] / "data" / "knowledge" / "endgame"

#: The most pieces (kings included) of a position the 3-4-5 piece tables cover.
MAX_PIECES = 5


class EndgameTablebase:
    """Syzygy tables opened from a directory.

    :param directory: Folder with the ``*.rtbw`` (win/draw/loss) and ``*.rtbz`` (distance to zeroing) files.
    :raises FileNotFoundError: If the folder does not exist.
    """

    def __init__(self, directory: Union[str, Path] = DEFAULT_TABLEBASE_DIR):
        path = Path(directory)
        if not path.is_dir():
            raise FileNotFoundError(f"tablebase directory not found: {path}")
        self.directory = path
        self._tables = _core.Tablebase.open(str(path))

    @classmethod
    def open_if_present(
        cls, directory: Union[str, Path] = DEFAULT_TABLEBASE_DIR
    ) -> Optional["EndgameTablebase"]:
        """Open the tables if the folder exists.

        :param directory: Folder with the tables.
        :return: The tablebase, or ``None`` when the folder is missing.
        """
        return cls(directory) if Path(directory).is_dir() else None

    @classmethod
    @lru_cache(maxsize=1)
    def shipped(cls) -> Optional["EndgameTablebase"]:
        """The tables of ``data/knowledge/endgame``, opened once and shared.

        :return: The tablebase, or ``None`` when the folder is missing (a fresh clone).
        """
        return cls.open_if_present()

    def close(self) -> None:
        """Release the open table files (they are closed when the object is dropped)."""
        self._tables = None

    def covers(self, board: chess.Board) -> bool:
        """Whether the position can be probed.

        :param board: Position to check.
        :return: True with at most :data:`MAX_PIECES` pieces and no castling rights.
        """
        return self._tables.covers(board.fen())

    def wdl(self, board: chess.Board) -> Optional[int]:
        """Result for the side to move, ignoring the halfmove clock.

        :param board: Position to probe.
        :return: 2 win, 1 win the 50-move rule spoils, 0 draw, -1 loss the rule saves, -2 loss; or
            ``None`` when a table is missing.
        """
        return self._tables.wdl(board.fen())

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Best move according to the tables.

        Moves are ranked by the result they keep (win, draw, loss, counting the 50-move rule),
        then by the distance to zeroing: a winning move is the one that wins soonest, a losing
        move the one that resists longest. A move that mates is always chosen first. Among equal
        moves the first in python-chess' order wins.

        :param board: Position to play from; it is not modified.
        :return: The move, or ``None`` when the position is not covered, has no legal move or a
            needed table is missing.
        """
        uci = self._tables.best_move(board.fen())
        return chess.Move.from_uci(uci) if uci else None
