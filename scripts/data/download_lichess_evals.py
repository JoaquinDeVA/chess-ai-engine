#!/usr/bin/env python3
"""Build ``data/evaluations.csv`` from the public Lichess evaluation database (CC0).

Only the beginning of the (multi-gigabyte) compressed file is read: the stream is closed as soon
as ``--rows`` positions have been converted.

    python scripts/data/download_lichess_evals.py --output data/evaluations.csv --rows 1000000
"""

import argparse
import io
import urllib.request
from pathlib import Path

import zstandard

from chess_engine.training.lichess import convert, write_csv

DEFAULT_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


def main() -> int:
    """Stream the database, convert the first rows and write the CSV.

    :return: Process exit code.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=Path("data/evaluations.csv"))
    parser.add_argument("--rows", type=int, default=1_000_000, help="positions to keep")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.url) as response:  # noqa: S310 (fixed public URL)
        stream = zstandard.ZstdDecompressor().stream_reader(response)
        lines = io.TextIOWrapper(stream, encoding="utf-8")
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            written = write_csv(convert(lines, args.rows), handle)
    print(f"Wrote {written:,} positions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
