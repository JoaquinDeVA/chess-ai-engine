"""Opening book: a mapping from positions to the move to play in them."""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import chess

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
        self._moves: Dict[str, str] = {}
        for fen, uci in entries.items():
            key = " ".join(fen.split()[:4])
            self._moves.setdefault(key, uci)

    @classmethod
    def load(cls, path: Union[str, Path] = DEFAULT_BOOK_PATH) -> "OpeningBook":
        """Load a book from a JSON file.

        :param path: JSON file mapping FEN strings to UCI moves.
        :return: The loaded book.
        :raises FileNotFoundError: If ``path`` does not exist.
        """
        with open(path, encoding="utf-8") as handle:
            return cls(json.load(handle))

    def lookup(self, board: chess.Board) -> Optional[chess.Move]:
        """Return the book move for ``board`` if there is one and it is legal.

        :param board: Current position.
        :return: The move, or ``None`` when the position is not in the book.
        """
        uci = self._moves.get(position_key(board))
        if uci is None:
            return None
        move = chess.Move.from_uci(uci)
        return move if move in board.legal_moves else None

    def __len__(self) -> int:
        return len(self._moves)
