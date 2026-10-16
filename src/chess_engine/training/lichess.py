"""Conversion of the public Lichess evaluation database to the project's CSV format.

The database (https://database.lichess.org, CC0) is a ``.jsonl.zst`` file with one position per
line: ``{"fen": ..., "evals": [{"depth": ..., "pvs": [{"cp": ...} | {"mate": ...}, ...]}]}``,
evaluated with Stockfish. Its documentation does not say whose point of view the numbers are
from, so :func:`detect_perspective` measures it from the data instead of assuming it.
"""

import csv
import io
import json
from collections.abc import Iterable, Iterator
from typing import Any

import chess
import numpy as np

CSV_COLUMNS = ("fen", "cp", "is_mate", "mate_distance")
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
WHITE, SIDE_TO_MOVE = "white", "side-to-move"
#: Records inspected to detect the perspective.
DETECTION_SAMPLE = 5000


def parse_record(line: str) -> dict[str, Any] | None:
    """Extract the deepest evaluation of a database line.

    :param line: One JSON line of the database.
    :return: ``{"fen", "cp", "mate"}`` (exactly one of ``cp``/``mate`` is not None), as written
        in the database, or None if the line has no usable evaluation.
    """
    try:
        record = json.loads(line)
        best = max(record["evals"], key=lambda evaluation: evaluation["depth"])
        principal = best["pvs"][0]
        return {"fen": record["fen"], "cp": principal.get("cp"), "mate": principal.get("mate")}
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def material_difference(fen: str) -> int:
    """White material minus Black material, in pawns (kings excluded).

    :param fen: Position.
    :return: The balance, positive when White has more.
    """
    board = chess.Board(fen)
    return sum(
        PIECE_VALUES.get(piece.piece_type, 0) * (1 if piece.color == chess.WHITE else -1)
        for piece in board.piece_map().values()
    )


def detect_perspective(records: Iterable[dict[str, Any]]) -> str:
    """Find out whether ``cp`` is from White's or from the side to move's point of view.

    Looks at positions where Black is to move: if ``cp`` grows with White's material advantage
    the numbers are from White's point of view, if it shrinks they are from Black's (the side to
    move's).

    :param records: Parsed records (see :func:`parse_record`).
    :return: :data:`WHITE` or :data:`SIDE_TO_MOVE`.
    :raises ValueError: If there is not enough data to decide.
    """
    cps, balances = [], []
    for record in records:
        if record["cp"] is None or " b " not in f" {record['fen']} ":
            continue
        cps.append(record["cp"])
        balances.append(material_difference(record["fen"]))
    if len(cps) < 50 or np.std(balances) == 0 or np.std(cps) == 0:
        raise ValueError("not enough Black-to-move positions to detect the perspective")
    correlation = np.corrcoef(cps, balances)[0, 1]
    return WHITE if correlation > 0 else SIDE_TO_MOVE


def to_row(record: dict[str, Any], perspective: str) -> dict[str, Any]:
    """Convert a record into a CSV row with evaluations from White's point of view.

    :param record: Parsed record.
    :param perspective: :data:`WHITE` or :data:`SIDE_TO_MOVE`.
    :return: Row with the :data:`CSV_COLUMNS` keys.
    """
    sign = -1 if perspective == SIDE_TO_MOVE and " b " in f" {record['fen']} " else 1
    is_mate = record["mate"] is not None
    return {
        "fen": record["fen"],
        "cp": "" if is_mate else sign * record["cp"],
        "is_mate": is_mate,
        "mate_distance": sign * record["mate"] if is_mate else "",
    }


def convert(lines: Iterable[str], limit: int) -> Iterator[dict[str, Any]]:
    """Convert the first ``limit`` usable lines to CSV rows (White's point of view).

    The perspective is detected on the first records (up to :data:`DETECTION_SAMPLE` of them,
    even if ``limit`` is smaller), then applied to all of them.

    :param lines: Lines of the database.
    :param limit: Number of rows to produce.
    :return: Iterator of rows.
    """
    records: list[dict[str, Any]] = []
    iterator = iter(lines)
    for line in iterator:
        record = parse_record(line)
        if record is not None:
            records.append(record)
        if len(records) >= DETECTION_SAMPLE:
            break
    perspective = detect_perspective(records)

    produced = 0
    for record in records:
        if produced >= limit:
            return
        yield to_row(record, perspective)
        produced += 1
    for line in iterator:
        if produced >= limit:
            return
        record = parse_record(line)
        if record is not None:
            yield to_row(record, perspective)
            produced += 1


def write_csv(rows: Iterable[dict[str, Any]], handle: io.TextIOBase) -> int:
    """Write rows with the project's CSV header.

    :param rows: Rows from :func:`convert`.
    :param handle: Text file opened for writing.
    :return: Number of rows written.
    """
    writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    count = 0
    for row in rows:
        writer.writerow(row)
        count += 1
    return count
