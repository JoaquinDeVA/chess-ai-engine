#!/usr/bin/env python3
"""Build the sidecar (split, quiet flag, side to move) of every raw shard that lacks one.

One shard per worker process; existing sidecars are skipped, so the run can be repeated.

    python src/chess_lab/scripts/data/build_sidecars.py --raw data/lichess/raw --out data/lichess/sidecar --jobs 5
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from chess_lab.training.sidecar import build_sidecar, sidecar_path


def main() -> None:
    """Parse the arguments and build the missing sidecars."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/lichess/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/lichess/sidecar"))
    parser.add_argument("--jobs", type=int, default=5, help="worker processes")
    args = parser.parse_args()

    shards = sorted(args.raw.glob("part-*.parquet"))
    todo = [s for s in shards if not sidecar_path(s, args.out).exists()]
    print(f"{len(shards)} shards, {len(shards) - len(todo)} already done, {len(todo)} to build")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(build_sidecar, shard, args.out): shard for shard in todo}
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            print(f"[{done}/{len(todo)}] {futures[future].name}", flush=True)


if __name__ == "__main__":
    main()
