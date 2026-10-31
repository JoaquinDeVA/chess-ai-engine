"""Raw (lossless) Parquet export of the public Lichess evaluation database.

One row per database line, with the deepest evaluation exactly as stored (no sign change). The
perspective of ``cp``/``mate`` (see :func:`chess_lab.training.lichess.detect_perspective`) is
recorded in the metadata of every Parquet file under the ``perspective`` key.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import chess
import pyarrow as pa
import pyarrow.parquet as pq

RAW_COLUMNS = (
    "fen",
    "cp",
    "mate",
    "depth",
    "knodes",
    "best_move",
    "n_evals",
    "n_pvs",
)
RAW_SCHEMA = pa.schema(
    [
        ("fen", pa.string()),
        ("cp", pa.int16()),
        ("mate", pa.int16()),
        ("depth", pa.int16()),
        ("knodes", pa.int32()),
        ("best_move", pa.string()),
        ("n_evals", pa.uint8()),
        ("n_pvs", pa.uint8()),
    ]
)
#: Rows per Parquet row group.
ROW_GROUP_SIZE = 100_000


def parse_raw_record(line: str) -> dict[str, Any] | None:
    """Extract the deepest evaluation of a database line, keeping its metadata.

    :param line: One JSON line of the database.
    :return: Dict with the :data:`RAW_COLUMNS` keys (``cp`` and ``mate`` as in the database,
        exactly one of them not None), or None if the line has no usable evaluation. Ties in
        depth are broken by ``knodes``.
    """
    try:
        record = json.loads(line)
        evals = record["evals"]
        best = max(evals, key=lambda e: (e["depth"], e.get("knodes", 0)))
        pvs = best["pvs"]
        principal = pvs[0]
        cp, mate = principal.get("cp"), principal.get("mate")
        if (cp is None) == (mate is None):
            return None
        moves = principal.get("line", "").split()
        return {
            "fen": record["fen"],
            "cp": cp,
            "mate": mate,
            "depth": best["depth"],
            "knodes": best.get("knodes", 0),
            "best_move": moves[0] if moves else None,
            "n_evals": min(len(evals), 255),
            "n_pvs": min(len(pvs), 255),
        }
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return None


def is_legal_position(fen: str) -> bool:
    """Tell whether a FEN is a legal standard-chess position (``chess.Board.is_valid``).

    :param fen: Position as stored in the database.
    :return: False for unparsable FENs and for positions such as Chess960 setups or boards with
        more than 32 pieces.
    """
    try:
        return chess.Board(fen).is_valid()
    except ValueError:
        return False


def parse_chunk(lines: list[str]) -> tuple[pa.Table, int, int]:
    """Parse lines and keep the legal positions (the unit of work of the parallel conversion).

    :param lines: Lines of the database.
    :return: Table of the kept records, number of unusable lines and number of lines whose
        position is not legal (see :func:`is_legal_position`).
    """
    records, unusable, illegal = [], 0, 0
    for line in lines:
        record = parse_raw_record(line)
        if record is None:
            unusable += 1
        elif is_legal_position(record["fen"]):
            records.append(record)
        else:
            illegal += 1
    return to_table(records), unusable, illegal


def parse_lines(lines: Iterable[str]) -> Iterator[dict[str, Any] | None]:
    """Parse every line; unusable ones yield None so the caller can count them.

    :param lines: Lines of the database.
    :return: One record or None per line.
    """
    for line in lines:
        yield parse_raw_record(line)


def to_table(records: list[dict[str, Any]]) -> pa.Table:
    """Build an Arrow table with :data:`RAW_SCHEMA` from parsed records.

    :param records: Records from :func:`parse_raw_record`.
    :return: The table (raises ``pyarrow.ArrowInvalid`` if a value overflows its dtype).
    """
    columns = {name: [r[name] for r in records] for name in RAW_COLUMNS}
    return pa.table(columns, schema=RAW_SCHEMA)


class ShardWriter:
    """Write one shard row group by row group (bounded memory), atomically.

    Data goes to a temporary file that is renamed on :meth:`close`, so a shard either exists
    complete or does not exist.
    """

    def __init__(self, path: Path):
        """Open the temporary file.

        :param path: Final ``.parquet`` path.
        """
        self.path = path
        self.temporary = path.with_suffix(".tmp")
        self.writer = pq.ParquetWriter(
            self.temporary, RAW_SCHEMA, compression="zstd", write_statistics=True
        )

    def write(self, records: list[dict[str, Any]]) -> None:
        """Append a row group.

        :param records: Records from :func:`parse_raw_record`.
        """
        if records:
            self.write_table(to_table(records))

    def write_table(self, table: pa.Table) -> None:
        """Append a table as row groups.

        :param table: Table with :data:`RAW_SCHEMA`.
        """
        if table.num_rows:
            self.writer.write_table(table, row_group_size=ROW_GROUP_SIZE)

    def close(self, metadata: dict[str, Any]) -> None:
        """Store the metadata, close the file and move it into place.

        :param metadata: JSON-serializable values stored in the file metadata.
        """
        self.writer.add_key_value_metadata({k: json.dumps(v) for k, v in metadata.items()})
        self.writer.close()
        self.temporary.replace(self.path)


def write_shard(path: Path, records: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Write a whole shard from memory (see :class:`ShardWriter`).

    :param path: Final ``.parquet`` path.
    :param records: Records of the shard.
    :param metadata: JSON-serializable values stored in the file metadata.
    """
    writer = ShardWriter(path)
    writer.write(records)
    writer.close(metadata)


def shard_path(directory: Path, index: int) -> Path:
    """Path of the ``index``-th shard.

    :param directory: Output directory.
    :param index: Shard number, from 0.
    :return: ``directory/part-00000.parquet``-style path.
    """
    return directory / f"part-{index:05d}.parquet"


def read_metadata(path: Path) -> dict[str, Any]:
    """Read the metadata written by :func:`write_shard`.

    :param path: Shard path.
    :return: The decoded metadata dict.
    """
    raw = pq.read_metadata(path).metadata or {}
    return {key.decode(): json.loads(value) for key, value in raw.items() if key != b"ARROW:schema"}
