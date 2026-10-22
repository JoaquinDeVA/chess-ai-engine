"""Endgame tablebases: perfect play in positions with few pieces.

A Syzygy tablebase stores, for every position with up to five pieces (kings included), whether the
side to move wins, draws or loses (WDL) and how many moves are left before the next capture or pawn
move (DTZ). With the 3-4-5 piece tables (about 940 MB) the engine never needs to search such a
position: :meth:`EndgameTablebase.best_move` returns a move that keeps the best result and, among
the winning moves, the one that finishes the quickest.

The tables are not versioned (they are large); see ``docs/DATA_AND_TRAINING.md`` for where they go.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import chess
import chess.syzygy

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
        self._tables = chess.syzygy.open_tablebase(str(path))

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
        """Release the open table files."""
        self._tables.close()

    def covers(self, board: chess.Board) -> bool:
        """Whether the position can be probed.

        :param board: Position to check.
        :return: True with at most :data:`MAX_PIECES` pieces and no castling rights.
        """
        return len(board.piece_map()) <= MAX_PIECES and not board.castling_rights

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Best move according to the tables.

        Moves are ranked by the result they keep (win, draw, loss, counting the 50-move rule),
        then by the distance to zeroing: a winning move is the one that wins soonest, a losing
        move the one that resists longest. A move that mates is always chosen first.

        :param board: Position to play from; it is not modified.
        :return: The move, or ``None`` when the position is not covered, has no legal move or a
            needed table is missing.
        """
        if not self.covers(board) or board.is_game_over():
            return None
        best_move, best_key = None, None
        probe = board.copy(stack=False)
        for move in list(board.legal_moves):
            probe.push(move)
            try:
                key = self._rank(probe)
            except (chess.syzygy.MissingTableError, KeyError):
                return None
            finally:
                probe.pop()
            if best_key is None or key > best_key:
                best_move, best_key = move, key
        return best_move

    def _rank(self, after: chess.Board) -> tuple:
        """Sort key (higher is better) of a position reached by our move, seen from the opponent."""
        if after.is_checkmate():
            return (3, 0)
        if after.is_stalemate() or after.is_insufficient_material():
            return (0, 0)
        wdl = -self._tables.probe_wdl(after)  # result for the side that just moved
        dtz = abs(self._tables.probe_dtz(after))
        if wdl > 0:
            return (2 if wdl == 2 else 1, -dtz)  # a plain win beats a win the 50-move rule spoils
        if wdl == 0:
            return (0, 0)
        return (-1 if wdl == -1 else -2, dtz)  # when losing, the longer the better
