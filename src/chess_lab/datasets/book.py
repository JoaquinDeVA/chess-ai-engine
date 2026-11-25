"""What the opening-book scripts share: the position key, the mate score, the book file and JSON lines."""

import json
from pathlib import Path
from typing import Dict, TextIO

import chess

#: Score of a forced mate, minus the moves it takes (a faster mate is worth more).
MATE_SCORE = 100_000.0


def fen_key(fen: str) -> str:
    """The first four FEN fields (placement, side to move, castling, en passant).

    The halfmove clock and the fullmove number are dropped so that transpositions hit the same
    entry, as in the ``uci`` binary.

    :param fen: Position in FEN, with four or more fields.
    :return: The key.
    """
    return " ".join(fen.split()[:4])


def position_key(board: chess.Board) -> str:
    """Key identifying a position independently of how it was reached.

    :param board: Position to identify.
    :return: Its :func:`fen_key`.
    """
    return fen_key(board.fen())


def load_book(path: Path) -> Dict[str, str]:
    """Read an opening book (a JSON ``{fen: uci}``) keyed by :func:`fen_key`.

    :param path: Book file.
    :return: ``{key: uci move}``.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {fen_key(fen): uci for fen, uci in raw.items()}


def read_jsonl(path: Path) -> list[dict]:
    """The records of a JSON-lines file; a line that does not parse (a cut write) is skipped.

    :param path: File to read.
    :return: The records, empty if the file does not exist.
    """
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def append_jsonl(handle: TextIO, record: dict) -> None:
    """Write one record as a line and flush it, so a cut loses at most that line.

    :param handle: File open for appending.
    :param record: JSON-serialisable record.
    """
    handle.write(json.dumps(record) + "\n")
    handle.flush()
