"""Opening book: a mapping from positions to the move to play in them.

The book itself lives in the Rust core (``rust/src/knowledge/book.rs``); this class is its Python
face.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Union

import chess

from chess_engine import _core

#: Default location of the book shipped with the project.
DEFAULT_BOOK_PATH = Path(__file__).resolve().parents[3] / "data" / "opening_book.json"


def position_key(board: chess.Board) -> str:
    """Key identifying a position independently of how it was reached.

    The halfmove clock and the fullmove number of the FEN are dropped so that transpositions
    (the same position reached through a different move order) hit the same entry.

    :param board: Position to identify.
    :return: The first four FEN fields (placement, side to move, castling, en passant).
    """
    return " ".join(board.fen().split()[:4])


class OpeningBook:
    """Read-only opening book loaded from a JSON file ``{fen: uci_move}``.

    Keys may be full FENs (move counters are ignored). If a position appears more than once
    the first entry wins.

    :param entries: Mapping from FEN to the move in UCI notation.
    """

    def __init__(self, entries: Dict[str, str]):
        self._book = _core.Book.from_entries(list(entries.items()))

    @classmethod
    def load(cls, path: Union[str, Path] = DEFAULT_BOOK_PATH) -> "OpeningBook":
        """Load a book from a JSON file.

        :param path: JSON file mapping FEN strings to UCI moves.
        :return: The loaded book.
        :raises FileNotFoundError: If ``path`` does not exist.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"opening book not found: {path}")
        book = cls({})
        book._book = _core.Book.load(str(path))
        return book

    @classmethod
    @lru_cache(maxsize=1)
    def shipped(cls) -> "OpeningBook":
        """The book shipped with the project, loaded once and shared.

        :return: The book of ``data/opening_book.json``; an empty book if the file is missing.
        """
        return cls.load() if DEFAULT_BOOK_PATH.exists() else cls({})

    def lookup(self, board: chess.Board) -> Optional[chess.Move]:
        """Return the book move for ``board`` if there is one and it is legal.

        :param board: Current position.
        :return: The move, or ``None`` when the position is not in the book.
        """
        uci = self._book.lookup(board.fen())
        return chess.Move.from_uci(uci) if uci else None

    def __len__(self) -> int:
        return len(self._book)
