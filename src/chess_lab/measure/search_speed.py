"""Time of the Rust engine to a fixed depth with 1, 2, 4... search threads.

``python -m chess_lab.measure.search_speed --threads 1,2,4,6``: for each thread count it runs
``go depth N`` on the first positions of the opening file and prints the mean seconds per position.
Run it on an otherwise idle machine; the numbers of a busy one mean nothing.
"""

import argparse
import statistics
import subprocess
import time
from pathlib import Path

from chess_engine.paths import ROOT

BINARY = ROOT / "rust" / "target" / "release" / "uci"
POSITIONS = ROOT / "data" / "openings" / "openings.epd"


def seconds_to_depth(threads: int, fens: list[str], depth: int, hash_mb: int) -> float:
    """Mean seconds the engine takes to finish depth ``depth`` on each position.

    :param threads: Value of the ``Threads`` option.
    :param fens: Positions to search.
    :param depth: Depth of every search.
    :param hash_mb: Value of the ``Hash`` option.
    :return: Mean wall-clock seconds per position.
    """
    engine = subprocess.Popen(
        [str(BINARY)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
    )
    assert engine.stdin is not None and engine.stdout is not None

    def send(line: str) -> None:
        engine.stdin.write(line + "\n")
        engine.stdin.flush()

    def wait_for(prefix: str) -> None:
        for line in engine.stdout:
            if line.startswith(prefix):
                return

    send(f"setoption name Threads value {threads}")
    send(f"setoption name Hash value {hash_mb}")
    send("isready")
    wait_for("readyok")
    times = []
    for fen in fens:
        send("ucinewgame")
        send(f"position fen {fen}")
        start = time.perf_counter()
        send(f"go depth {depth}")
        wait_for("bestmove")
        times.append(time.perf_counter() - start)
    send("quit")
    engine.wait()
    return statistics.fmean(times)


def main() -> None:
    """Print the table of mean seconds to depth for each thread count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", default="1,2,4,6", help="thread counts, comma separated")
    parser.add_argument("--positions", type=int, default=60)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--hash", type=int, default=256, help="megabytes of the shared table")
    args = parser.parse_args()
    lines = Path(POSITIONS).read_text().splitlines()[: args.positions]
    fens = [" ".join(line.split()[:4]) + " 0 1" for line in lines]
    print("| threads | seconds to depth", args.depth, "|")
    print("|---|---|")
    for threads in (int(t) for t in args.threads.split(",")):
        print(f"| {threads} | {seconds_to_depth(threads, fens, args.depth, args.hash):.3f} |")


if __name__ == "__main__":
    main()
