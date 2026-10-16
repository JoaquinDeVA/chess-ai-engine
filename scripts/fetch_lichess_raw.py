#!/usr/bin/env python3
"""Stream the public Lichess evaluation database (CC0) into sharded raw Parquet files.

The compressed file is never stored: it is decompressed on the fly and the connection is closed
after ``--rows`` database lines. Output is split in shards of ``--shard-rows`` lines
(``part-00000.parquet``, ...); shards that already exist are skipped (their lines are still
decompressed, not parsed), so an interrupted run can be repeated with the same options.

    python scripts/fetch_lichess_raw.py --rows 1000000 --output-dir data/lichess/raw
"""

import argparse
import io
import resource
import sys
import time
import urllib.request
from pathlib import Path

import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chess_engine.training.lichess import DETECTION_SAMPLE, detect_perspective  # noqa: E402
from chess_engine.training.lichess_raw import (  # noqa: E402
    ROW_GROUP_SIZE,
    ShardWriter,
    parse_raw_record,
    read_metadata,
    shard_path,
)

DEFAULT_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
PROGRESS_EVERY = 200_000


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


def fetch(url: str, rows: int, output_dir: Path, shard_rows: int) -> dict:
    """Run the download and return the measurements.

    :param url: Database URL.
    :param rows: Database lines to read (the first ones).
    :param output_dir: Directory for the shards.
    :param shard_rows: Database lines per shard.
    :return: Measurement summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start, cpu_start = time.perf_counter(), time.process_time()
    kept = dropped = lines_read = 0
    perspective = None
    existing = shard_path(output_dir, 0)
    if existing.exists():
        perspective = read_metadata(existing).get("perspective")

    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (given by the user)
        counter = CountingReader(response)
        reader = zstandard.ZstdDecompressor().stream_reader(io.BufferedReader(counter))
        lines = io.TextIOWrapper(reader, encoding="utf-8")
        index = 0
        while lines_read < rows:
            expected = min(shard_rows, rows - lines_read)
            path = shard_path(output_dir, index)
            if path.exists() and read_metadata(path).get("lines") == expected:
                meta = read_metadata(path)
                for _ in zip(range(expected), lines, strict=False):
                    pass
                kept, dropped = kept + meta["kept"], dropped + meta["dropped"]
                lines_read += expected
                index += 1
                continue
            writer, records, bad, count, saved = ShardWriter(path), [], 0, 0, 0
            for line in lines:
                count += 1
                record = parse_raw_record(line)
                if record is None:
                    bad += 1
                else:
                    records.append(record)
                if perspective is None and len(records) == DETECTION_SAMPLE:
                    perspective = detect_perspective(records)
                if len(records) >= ROW_GROUP_SIZE:
                    writer.write(records)
                    saved, records = saved + len(records), []
                if count >= expected:
                    break
                if (lines_read + count) % PROGRESS_EVERY == 0:
                    elapsed = time.perf_counter() - start
                    print(
                        f"{lines_read + count:>12,} lines  {counter.count / 1e6:8.1f} MB  "
                        f"{(lines_read + count) / elapsed:8.0f} lines/s",
                        flush=True,
                    )
            if perspective is None:
                perspective = detect_perspective(records)
            writer.write(records)
            saved += len(records)
            meta = {"perspective": perspective, "lines": count, "kept": saved, "dropped": bad}
            writer.close(meta)
            kept, dropped = kept + saved, dropped + bad
            lines_read += count
            index += 1
            if count < expected:
                break
    wall, cpu = time.perf_counter() - start, time.process_time() - cpu_start
    size = sum(p.stat().st_size for p in output_dir.glob("part-*.parquet"))
    return {
        "lines": lines_read,
        "kept": kept,
        "dropped": dropped,
        "perspective": perspective,
        "wall_s": wall,
        "cpu_s": cpu,
        "compressed_mb": counter.count / 1e6,
        "parquet_bytes": size,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def main() -> int:
    """Parse the options, run the download and print the measurements.

    :return: Process exit code.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--rows", type=int, default=1_000_000, help="database lines to read")
    parser.add_argument("--output-dir", type=Path, default=Path("data/lichess/raw"))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--shard-rows", type=int, default=1_000_000, help="lines per shard")
    args = parser.parse_args()

    m = fetch(args.url, args.rows, args.output_dir, args.shard_rows)
    print("\n== Measurements ==")
    print(f"lines read / kept / dropped : {m['lines']:,} / {m['kept']:,} / {m['dropped']:,}")
    print(f"perspective of cp and mate  : {m['perspective']}")
    print(f"wall time                   : {m['wall_s']:.1f} s  (cpu {m['cpu_s']:.1f} s)")
    print(f"compressed downloaded       : {m['compressed_mb']:.1f} MB")
    print(f"network throughput          : {m['compressed_mb'] / m['wall_s']:.2f} MB/s")
    print(f"parse rate                  : {m['lines'] / m['wall_s']:,.0f} lines/s")
    print(f"parquet size                : {m['parquet_bytes'] / 1e6:.1f} MB")
    print(f"bytes per position          : {m['parquet_bytes'] / max(m['kept'], 1):.1f}")
    print(f"peak memory                 : {m['peak_rss_mb']:.0f} MB")
    if m["lines"] < args.rows:
        print(f"WARNING: the stream ended after {m['lines']:,} lines; repeat to resume.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
