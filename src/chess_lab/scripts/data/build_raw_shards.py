#!/usr/bin/env python3
"""Convert the public Lichess evaluation database (CC0) into sharded raw Parquet files.

The input is the compressed file, either a local ``.zst`` (``--input``, e.g. downloaded with
``aria2c``) or a URL streamed over HTTP (``--url``). It is never stored by this script: it is
decompressed on the fly. Without ``--rows`` the whole file is converted. Output is split in
shards of ``--shard-rows`` lines (``part-00000.parquet``, ...); shards that already exist are
skipped (their lines are still decompressed, not parsed), so an interrupted run can be repeated
with the same options. Only legal standard-chess positions are kept (``chess.Board.is_valid``);
parsing and the legality check run in ``--workers`` processes.

    python src/chess_lab/scripts/data/build_raw_shards.py --input data/lichess/lichess_db_eval.jsonl.zst
"""

import argparse
import io
import itertools
import multiprocessing
import os
import resource
import time
import urllib.request
from collections import deque
from pathlib import Path

import pyarrow as pa
import zstandard

from chess_lab.training.lichess_raw import (
    DETECTION_SAMPLE,
    ROW_GROUP_SIZE,
    ShardWriter,
    detect_perspective,
    parse_chunk,
    read_metadata,
    shard_path,
)

DEFAULT_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
#: Database lines per unit of work sent to a worker process.
CHUNK_LINES = 20_000
#: Seconds between progress lines.
PROGRESS_SECONDS = 60


class CountingReader(io.RawIOBase):
    """Wrap a binary stream and count the bytes read from it."""

    def __init__(self, stream):
        """Wrap ``stream``.

        :param stream: Binary file-like object.
        """
        self.stream = stream
        self.count = 0

    def readable(self) -> bool:
        """Report that the stream can be read.

        :return: Always True.
        """
        return True

    def readinto(self, buffer) -> int:
        """Read into ``buffer`` and count the bytes.

        :param buffer: Writable bytes-like object.
        :return: Number of bytes read.
        """
        read = self.stream.readinto(buffer)
        self.count += read or 0
        return read


def open_source(source: str):
    """Open a local file or a URL for binary reading.

    :param source: Path of a local file or ``http(s)``/``file`` URL.
    :return: Binary stream and its size in bytes (0 if unknown).
    """
    if "://" not in source:
        return open(source, "rb"), os.path.getsize(source)  # noqa: SIM115 (closed by the caller)
    response = urllib.request.urlopen(source, timeout=60)  # noqa: S310 (given by the user)
    return response, int(response.headers.get("Content-Length") or 0)


def read_chunks(lines, total: int):
    """Cut the next ``total`` lines (or the rest of the stream) in chunks.

    :param lines: Iterator of text lines.
    :param total: Maximum number of lines.
    :return: Generator of lists of lines.
    """
    while total > 0:
        chunk = list(itertools.islice(lines, min(CHUNK_LINES, total)))
        if not chunk:
            return
        total -= len(chunk)
        yield chunk


def parse_in_order(pool, chunks, window: int):
    """Parse chunks keeping the order, with at most ``window`` chunks in flight.

    :param pool: Process pool, or None to parse in this process.
    :param chunks: Iterable of lists of lines.
    :param window: Maximum number of chunks being parsed (bounds the memory).
    :return: Generator of :func:`~chess_lab.training.lichess_raw.parse_chunk` results.
    """
    pending = deque()
    for chunk in chunks:
        if pool is None:
            yield parse_chunk(chunk)
            continue
        pending.append(pool.apply_async(parse_chunk, (chunk,)))
        if len(pending) >= window:
            yield pending.popleft().get()
    while pending:
        yield pending.popleft().get()


def format_duration(seconds: float) -> str:
    """Format a duration as ``H:MM:SS``.

    :param seconds: Duration.
    :return: The text.
    """
    return f"{int(seconds) // 3600}:{int(seconds) // 60 % 60:02d}:{int(seconds) % 60:02d}"


def fetch(
    source: str,
    rows: int | None,
    output_dir: Path,
    shard_rows: int,
    workers: int = 1,
    log_file: Path | None = None,
    progress_seconds: float = PROGRESS_SECONDS,
) -> dict:
    """Run the conversion and return the measurements.

    :param source: Path of a local ``.zst`` file or URL of the database.
    :param rows: Database lines to read (the first ones), or None for the whole file.
    :param output_dir: Directory for the shards.
    :param shard_rows: Database lines per shard.
    :param workers: Parsing processes (1 parses in this process).
    :param log_file: File where the progress lines are also appended.
    :param progress_seconds: Seconds between progress lines.
    :return: Measurement summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start, cpu_start = time.perf_counter(), time.process_time()
    kept = dropped = illegal = lines_read = 0
    perspective = None
    existing = shard_path(output_dir, 0)
    if existing.exists():
        perspective = read_metadata(existing).get("perspective")
    last_report = start

    def report(force: bool = False) -> None:
        nonlocal last_report
        now = time.perf_counter()
        if not force and now - last_report < progress_seconds:
            return
        last_report, elapsed = now, now - start
        done = counter.count
        rate = done / elapsed
        eta = format_duration((total_bytes - done) / rate) if total_bytes and rate else "?"
        text = (
            f"{lines_read:>12,} lines  {done / 1e9:7.2f}/{total_bytes / 1e9:.2f} GB "
            f"({100 * done / max(total_bytes, 1):5.1f}%)  {rate / 1e6:6.2f} MB/s  "
            f"{lines_read / elapsed:8.0f} lines/s  elapsed {format_duration(elapsed)}  "
            f"ETA {eta}  kept {kept:,}  unusable {dropped:,}  illegal {illegal:,}"
        )
        print(text, flush=True)
        if log_file is not None:
            with log_file.open("a") as handle:
                handle.write(text + "\n")

    raw, total_bytes = open_source(source)
    pool = multiprocessing.get_context("spawn").Pool(workers) if workers > 1 else None
    try:
        counter = CountingReader(raw)
        reader = zstandard.ZstdDecompressor().stream_reader(io.BufferedReader(counter))
        lines = io.TextIOWrapper(reader, encoding="utf-8")
        index = 0
        while rows is None or lines_read < rows:
            expected = shard_rows if rows is None else min(shard_rows, rows - lines_read)
            path = shard_path(output_dir, index)
            if path.exists() and read_metadata(path).get("lines") == expected:
                meta = read_metadata(path)
                for _ in zip(range(expected), lines, strict=False):
                    pass
                kept, dropped = kept + meta["kept"], dropped + meta["dropped"]
                illegal += meta.get("illegal", 0)
                lines_read += expected
                index += 1
                report()
                continue
            writer, tables, buffered, sample = ShardWriter(path), [], 0, []
            count = bad = bad_position = saved = 0
            chunks = read_chunks(lines, expected)
            for table, unusable, not_legal in parse_in_order(pool, chunks, 2 * workers):
                count += table.num_rows + unusable + not_legal
                bad, bad_position = bad + unusable, bad_position + not_legal
                tables.append(table)
                buffered += table.num_rows
                if perspective is None:
                    sample += table.to_pylist()
                    if len(sample) >= DETECTION_SAMPLE:
                        perspective = detect_perspective(sample[:DETECTION_SAMPLE])
                if buffered >= ROW_GROUP_SIZE:
                    writer.write_table(pa.concat_tables(tables))
                    saved, tables, buffered = saved + buffered, [], 0
                lines_read += table.num_rows + unusable + not_legal
                report()
            if perspective is None:
                perspective = detect_perspective(sample)
            if tables:
                writer.write_table(pa.concat_tables(tables))
            saved += buffered
            meta = {
                "perspective": perspective,
                "lines": count,
                "kept": saved,
                "dropped": bad,
                "illegal": bad_position,
            }
            writer.close(meta)
            kept, dropped, illegal = kept + saved, dropped + bad, illegal + bad_position
            index += 1
            if count < expected:
                break
    finally:
        if pool is not None:
            pool.terminate()
        raw.close()
    report(force=True)
    wall, cpu = time.perf_counter() - start, time.process_time() - cpu_start
    size = sum(p.stat().st_size for p in output_dir.glob("part-*.parquet"))
    return {
        "lines": lines_read,
        "kept": kept,
        "dropped": dropped,
        "illegal": illegal,
        "perspective": perspective,
        "wall_s": wall,
        "cpu_s": cpu,
        "compressed_mb": counter.count / 1e6,
        "parquet_bytes": size,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def main() -> int:
    """Parse the options, run the conversion and print the measurements.

    :return: Process exit code.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--rows", type=int, default=None, help="lines to read (default: all)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/lichess/raw"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", help="local .zst file")
    source.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--shard-rows", type=int, default=1_000_000, help="lines per shard")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 2),
        help="parsing processes (default: cores - 2)",
    )
    parser.add_argument("--log-file", type=Path, default=None, help="also append the progress")
    parser.add_argument("--progress-seconds", type=float, default=PROGRESS_SECONDS)
    args = parser.parse_args()

    m = fetch(
        args.input or args.url,
        args.rows,
        args.output_dir,
        args.shard_rows,
        args.workers,
        args.log_file,
        args.progress_seconds,
    )
    print("\n== Measurements ==")
    print(f"lines read                  : {m['lines']:,}")
    print(f"kept / unusable / illegal   : {m['kept']:,} / {m['dropped']:,} / {m['illegal']:,}")
    print(f"perspective of cp and mate  : {m['perspective']}")
    print(
        f"wall time                   : {m['wall_s']:.1f} s  (main process cpu {m['cpu_s']:.1f} s)"
    )
    print(f"compressed read             : {m['compressed_mb']:.1f} MB")
    print(f"input throughput            : {m['compressed_mb'] / m['wall_s']:.2f} MB/s")
    print(f"parse rate                  : {m['lines'] / m['wall_s']:,.0f} lines/s")
    print(f"parquet size                : {m['parquet_bytes'] / 1e6:.1f} MB")
    print(f"bytes per position          : {m['parquet_bytes'] / max(m['kept'], 1):.1f}")
    print(f"peak memory (main process)  : {m['peak_rss_mb']:.0f} MB")
    if args.rows is not None and m["lines"] < args.rows:
        print(f"WARNING: the stream ended after {m['lines']:,} lines; repeat to resume.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
