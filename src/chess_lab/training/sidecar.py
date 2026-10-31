"""Sidecar of the raw shards (split, quiet flag, side to move) and samplers over them.

The sidecar of ``part-NNNNN.parquet`` is ``part-NNNNN.sidecar.parquet``: same rows in the same
order and the same row-group layout, so a row group of the shard pairs with the same row group of
its sidecar and the samplers read only what they need.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import chess
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from chess_lab.models.representation import is_quiet, keep_mate_rows, validate_mate_keep

from .dataset import assign_split

SIDECAR_SCHEMA = pa.schema(
    [("split", pa.int8()), ("quiet", pa.bool_()), ("white_to_move", pa.bool_())]
)
#: Columns returned by the samplers; ``depth`` is the raw search depth of the evaluation.
SAMPLE_COLUMNS = ["fen", "cp", "mate", "best_move", "quiet", "white_to_move", "depth"]
_RAW_COLUMNS = ["fen", "cp", "mate", "best_move", "depth"]


def sidecar_path(shard: Path, out_dir: Path) -> Path:
    """Path of the sidecar of a shard.

    :param shard: Raw shard path.
    :param out_dir: Sidecar directory.
    :return: ``out_dir/<shard stem>.sidecar.parquet``.
    """
    return out_dir / f"{shard.stem}.sidecar.parquet"


def build_sidecar(shard: Path, out_dir: Path) -> Path:
    """Compute the sidecar of a shard, streaming it by row group, atomically.

    :param shard: Raw shard.
    :param out_dir: Sidecar directory (created if missing).
    :return: Path of the sidecar, with the same rows and row groups as the shard.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = sidecar_path(shard, out_dir)
    temporary = path.with_suffix(".tmp")
    source = pq.ParquetFile(shard)
    rows = 0
    with pq.ParquetWriter(temporary, SIDECAR_SCHEMA, compression="zstd") as writer:
        for group in range(source.num_row_groups):
            table = source.read_row_group(group, columns=["fen", "best_move"])
            fens, moves = table["fen"].to_pylist(), table["best_move"].to_pylist()
            boards = [chess.Board(fen) for fen in fens]
            writer.write_table(
                pa.table(
                    {
                        "split": assign_split(pd.Series(fens)),
                        "quiet": [is_quiet(b, m) for b, m in zip(boards, moves, strict=True)],
                        "white_to_move": [b.turn == chess.WHITE for b in boards],
                    },
                    schema=SIDECAR_SCHEMA,
                )
            )
            rows += table.num_rows
        writer.add_key_value_metadata({"rows": json.dumps(rows), "source": json.dumps(shard.name)})
    temporary.replace(path)
    return path


def _open_pairs(raw_dir: Path, sidecar_dir: Path) -> list[tuple[pq.ParquetFile, pq.ParquetFile]]:
    """Open every shard with its sidecar, checking that they are aligned.

    :raises ValueError: If a sidecar is missing or its rows or row groups differ from the shard's.
    """
    pairs = []
    for shard in sorted(raw_dir.glob("part-*.parquet")):
        side = sidecar_path(shard, sidecar_dir)
        if not side.exists():
            raise ValueError(f"missing sidecar for {shard.name}: {side}")
        raw_file, side_file = pq.ParquetFile(shard), pq.ParquetFile(side)
        raw_meta, side_meta = raw_file.metadata, side_file.metadata
        layout = [raw_meta.row_group(i).num_rows for i in range(raw_meta.num_row_groups)]
        side_layout = [side_meta.row_group(i).num_rows for i in range(side_meta.num_row_groups)]
        if layout != side_layout:
            raise ValueError(
                f"{shard.name}: sidecar has {side_meta.num_rows} rows, shard {raw_meta.num_rows}"
            )
        pairs.append((raw_file, side_file))
    return pairs


def _kept_mates(fens: np.ndarray, is_mate: np.ndarray, mate_keep: float) -> np.ndarray:
    """Rows kept by the mate undersampling: all non-mates, and the mates of :func:`keep_mate_rows`.

    :param fens: FENs of the rows.
    :param is_mate: Boolean array, True on the mate rows.
    :param mate_keep: Share of the mates to keep, in ``(0, 1]``.
    :return: Boolean array, one entry per row.
    """
    keep = np.ones(len(fens), dtype=bool)
    if mate_keep < 1.0:
        mates = np.flatnonzero(is_mate)
        keep[mates] = keep_mate_rows(fens[mates], mate_keep)
    return keep


def _eligible(
    pair: tuple[pq.ParquetFile, pq.ParquetFile], split: int, quiet_only: bool, mate_keep: float
) -> np.ndarray:
    """Row indices of a shard in ``split``, quiet if requested, that survive the mate sampling.

    :param pair: Raw shard and its sidecar.
    :param split: ``TRAIN``, ``VALIDATION`` or ``TEST``.
    :param quiet_only: Keep only quiet positions.
    :param mate_keep: Share of the mate positions kept.
    :return: Row indices.
    """
    raw_file, side_file = pair
    table = side_file.read(columns=["split", "quiet"])
    mask = table["split"].to_numpy() == split
    if quiet_only:
        mask &= table["quiet"].to_numpy(zero_copy_only=False)
    if mate_keep < 1.0:
        raw = raw_file.read(columns=["fen", "mate"]).to_pandas()
        is_mate = raw["mate"].notna().to_numpy()
        mask &= _kept_mates(raw["fen"].to_numpy(dtype=object), is_mate, mate_keep)
    return np.flatnonzero(mask)


def _read_group(pair: tuple[pq.ParquetFile, pq.ParquetFile], group: int) -> pd.DataFrame:
    """One row group of a shard joined with its sidecar columns."""
    raw_file, side_file = pair
    frame = raw_file.read_row_group(group, columns=_RAW_COLUMNS).to_pandas()
    side = side_file.read_row_group(group, columns=["split", "quiet", "white_to_move"])
    side = side.to_pandas()
    frame["split"] = side["split"]
    frame["quiet"] = side["quiet"]
    frame["white_to_move"] = side["white_to_move"]
    return frame


def _group_bounds(raw_file: pq.ParquetFile) -> np.ndarray:
    """Cumulative row counts of the row groups, starting at 0."""
    meta = raw_file.metadata
    sizes = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
    return np.concatenate([[0], np.cumsum(sizes)])


def load_sample(
    raw_dir: Path,
    sidecar_dir: Path,
    split: int,
    n: int,
    seed: int = 42,
    quiet_only: bool = False,
    mate_keep: float = 1.0,
) -> pd.DataFrame:
    """Uniform random sample of ``n`` positions of a split, without loading whole shards.

    Memory scales with one shard (the per-shard sample sizes are drawn from eligible counts). The
    sampling is uniform by row, so most row groups are read once per call: the cost is I/O.

    :param raw_dir: Directory with the raw shards.
    :param sidecar_dir: Directory with their sidecars.
    :param split: ``TRAIN``, ``VALIDATION`` or ``TEST``.
    :param n: Rows wanted (fewer are returned if the split has fewer).
    :param seed: Seed of the sample; the same seed gives the same rows.
    :param quiet_only: Sample only quiet positions.
    :param mate_keep: Share of the mate positions that are eligible (see
        :func:`~.representation.keep_mate_rows`); ``n`` counts the rows after this filter.
    :return: DataFrame with columns :data:`SAMPLE_COLUMNS`.
    :raises ValueError: If a sidecar is missing or misaligned with its shard, or ``mate_keep``
        is outside ``(0, 1]``.
    """
    validate_mate_keep(mate_keep)
    pairs = _open_pairs(raw_dir, sidecar_dir)
    sizes = np.array(
        [len(_eligible(pair, split, quiet_only, mate_keep)) for pair in pairs], dtype=np.int64
    )
    total = int(sizes.sum())
    rng = np.random.default_rng(seed)
    wanted = rng.multivariate_hypergeometric(sizes, min(n, total)) if total else sizes
    parts = []
    for pair, count in zip(pairs, wanted, strict=True):
        if not count:
            continue
        rows = _eligible(pair, split, quiet_only, mate_keep)
        local = np.sort(rng.choice(rows, size=int(count), replace=False))
        bounds = _group_bounds(pair[0])
        groups = np.searchsorted(bounds, local, side="right") - 1
        for group in np.unique(groups):
            frame = _read_group(pair, int(group))
            parts.append(frame.iloc[local[groups == group] - bounds[group]])
    if not parts:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    frame = pd.concat(parts, ignore_index=True)
    return frame.iloc[rng.permutation(len(frame))][SAMPLE_COLUMNS].reset_index(drop=True)


def iter_batches(
    raw_dir: Path,
    sidecar_dir: Path,
    split: int,
    batch_size: int,
    n: int | None,
    seed: int,
    quiet_only: bool = False,
    mate_keep: float = 1.0,
) -> Iterator[pd.DataFrame]:
    """Stream shuffled batches of a split (shuffled shard order, shuffled within row groups).

    :param raw_dir: Directory with the raw shards.
    :param sidecar_dir: Directory with their sidecars.
    :param split: ``TRAIN``, ``VALIDATION`` or ``TEST``.
    :param batch_size: Rows per batch (the last one may be smaller).
    :param n: Stop after this many rows in total, or None for one pass over the split.
    :param seed: Seed of the shuffling.
    :param quiet_only: Yield only quiet positions.
    :param mate_keep: Share of the mate positions yielded (see :func:`load_sample`).
    :return: Iterator of DataFrames with columns :data:`SAMPLE_COLUMNS`.
    :raises ValueError: If a sidecar is missing or misaligned with its shard, or ``mate_keep``
        is outside ``(0, 1]``.
    """
    validate_mate_keep(mate_keep)
    pairs = _open_pairs(raw_dir, sidecar_dir)
    rng = np.random.default_rng(seed)
    if n == 0:
        return
    remaining = n
    buffer = pd.DataFrame(columns=SAMPLE_COLUMNS)
    for shard_index in rng.permutation(len(pairs)):
        pair = pairs[shard_index]
        for group in rng.permutation(pair[0].num_row_groups):
            frame = _read_group(pair, int(group))
            keep = frame["split"] == split
            if quiet_only:
                keep &= frame["quiet"]
            if mate_keep < 1.0:
                keep &= _kept_mates(
                    frame["fen"].to_numpy(dtype=object),
                    frame["mate"].notna().to_numpy(),
                    mate_keep,
                )
            frame = frame[keep]
            frame = frame.iloc[rng.permutation(len(frame))][SAMPLE_COLUMNS]
            buffer = pd.concat([buffer, frame], ignore_index=True)
            while len(buffer) >= batch_size:
                batch, buffer = buffer.iloc[:batch_size], buffer.iloc[batch_size:]
                if remaining is not None:
                    batch = batch.iloc[:remaining]
                    remaining -= len(batch)
                yield batch.reset_index(drop=True)
                if remaining == 0:
                    return
    if len(buffer):
        yield buffer.iloc[:remaining].reset_index(drop=True)
