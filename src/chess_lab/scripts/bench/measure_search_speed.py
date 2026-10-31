#!/usr/bin/env python3
"""Measure how deep and how fast the engine searches.

python src/chess_lab/scripts/bench/measure_search_speed.py --positions 200 --seconds 1,3,10 --depths 4
python src/chess_lab/scripts/bench/measure_search_speed.py --label python-engine --out data/reports/SPEED_python.json

Every position of a fixed sample of the test split (seed 42) is searched by a fresh player, in two
ways:

* for a time budget (``--seconds``): the depth reached in that time;
* to a fixed depth (``--depths``): the time and the nodes it takes.

For each player and setting the report gives the mean and median completed depth, the mean seconds
per move and the nodes per second. Players are the heuristic ``h1`` and the registered models;
books and tablebases are off, as in the tournaments. The jobs (one per player and setting) run in
parallel processes, each with one thread; use the same ``--workers`` to compare two engines.

Writes the JSON results to ``--out`` and a Markdown table next to it. Nothing is written to the
model cards.
"""

import argparse
import json
import multiprocessing
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import chess

DEFAULT_PLAYERS = "h1,linear_material,linear_pst,linear_pst_terms,linear_king_bucket,net4_pst"
#: Clock that never limits a fixed-depth search.
GENEROUS_CLOCK = 1e6
SINGLE_THREAD = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_THREADS")


def load_fens(raw: Path, sidecar: Path, count: int) -> List[str]:
    """The positions of the sample.

    :param raw: Folder with the shards.
    :param sidecar: Folder with the sidecars.
    :param count: Number of positions.
    :return: FEN strings.
    """
    from chess_lab.comparison.evaluate import SAMPLE_SEED, TEST
    from chess_lab.training.sidecar import load_sample

    return load_sample(raw, sidecar, TEST, count, seed=SAMPLE_SEED)["fen"].tolist()


def run_job(job: Tuple[str, str, float, List[str], int, Tuple[str, ...]]) -> Dict[str, Any]:
    """Search every position with one player and one setting.

    :param job: ``(player, kind, value, fens, threads, techniques)`` with ``kind`` ``"seconds"`` or ``"depth"``.
    :return: The player, the setting and the summary of the searches.
    """
    from chess_engine.bots.time_manager import TimeManager
    from chess_lab.comparison.tradeoff import fixed_time_manager, make_player

    player_name, kind, value, fens, threads, techniques = job
    options = {} if threads == 1 else {"threads": threads, "hash_mb": 256}
    options["techniques"] = techniques
    if kind == "seconds":
        player = make_player(player_name, fixed_time_manager(value), **options)
        clock = value
    else:
        player = make_player(player_name, TimeManager(), max_depth=int(value), **options)
        clock = GENEROUS_CLOCK
    depths: List[int] = []
    seconds: List[float] = []
    nodes = 0
    for fen in fens:
        board = chess.Board(fen)
        player.on_game_start(board.turn, "speed")  # fresh table for every position
        started = time.perf_counter()
        player.get_move(board, clock)
        seconds.append(time.perf_counter() - started)
        depths.append(player.last_search_stats.completed_depth)
        nodes += player.last_search_stats.nodes_visited
    return {
        "player": player_name,
        "threads": threads,
        "techniques": list(techniques),
        "kind": kind,
        "value": value,
        "positions": len(fens),
        "mean_depth": statistics.fmean(depths),
        "median_depth": statistics.median(depths),
        "mean_seconds": statistics.fmean(seconds),
        "nodes_per_second": nodes / max(sum(seconds), 1e-9),
    }


def render(results: Sequence[Dict[str, Any]], label: str) -> str:
    """The results as Markdown tables.

    :param results: Output of :func:`run_job` for every job.
    :param label: Name of the engine measured.
    :return: Markdown text.
    """
    lines = [f"# Search speed: {label}", ""]
    many = any(r.get("threads", 1) != 1 for r in results)
    for kind, title, unit in (
        ("seconds", "Depth reached in a time budget", "Seconds"),
        ("depth", "Cost of a fixed depth", "Depth"),
    ):
        rows = [r for r in results if r["kind"] == kind]
        if not rows:
            continue
        head = f"| Player | {'Threads | ' if many else ''}{unit} | Mean depth | Median depth | Mean s/move | Nodes/s |"
        lines += [f"## {title}", "", head, "|---|" + "---|" * (6 + many)]
        for r in rows:
            threads = f"{r.get('threads', 1)} | " if many else ""
            lines.append(
                f"| `{r['player']}` | {threads}{r['value']:g} | {r['mean_depth']:.2f} | "
                f"{r['median_depth']:g} | {r['mean_seconds']:.3f} | {r['nodes_per_second']:,.0f} |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", default=DEFAULT_PLAYERS)
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--seconds", default="1,3,10", help="time budgets in seconds, or empty")
    parser.add_argument("--depths", default="4", help="fixed depths, or empty")
    parser.add_argument("--workers", type=int, default=5, help="parallel processes")
    parser.add_argument(
        "--techniques",
        default="",
        help="techniques of selective search on (comma-separated, or all)",
    )
    parser.add_argument(
        "--threads",
        default="1",
        help="threads of the search, comma-separated (use --workers 1 when above 1)",
    )
    parser.add_argument("--label", default="engine", help="name of the engine measured")
    parser.add_argument("--raw", type=Path, default=Path("data/lichess/raw"))
    parser.add_argument("--sidecar", type=Path, default=Path("data/lichess/sidecar"))
    parser.add_argument("--out", type=Path, default=Path("data/reports/SPEED.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the jobs and write the results.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    fens = load_fens(args.raw, args.sidecar, args.positions)
    players = [p for p in args.players.split(",") if p]
    threads = [int(n) for n in args.threads.split(",")]
    from chess_engine import _core

    names = list(_core.TECHNIQUES) if args.techniques == "all" else args.techniques.split(",")
    techniques = tuple(n for n in names if n)
    jobs = [
        (p, "seconds", float(s), fens, n, techniques)
        for p in players
        for n in threads
        for s in args.seconds.split(",")
        if s
    ]
    jobs += [
        (p, "depth", float(d), fens, n, techniques)
        for p in players
        for n in threads
        for d in args.depths.split(",")
        if d
    ]
    for name in SINGLE_THREAD:
        os.environ[name] = "1"
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        results = list(pool.map(run_job, jobs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"label": args.label, "results": results}, indent=1) + "\n")
    text = render(results, args.label)
    args.out.with_suffix(".md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
