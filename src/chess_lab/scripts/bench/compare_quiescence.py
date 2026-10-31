#!/usr/bin/env python3
"""Measure what the quiescence search adds at a fixed depth.

python src/chess_lab/scripts/bench/compare_quiescence.py --players h1,linear_pst_terms --positions 1000

Every position of a fixed sample of the test split (seed 42) is searched at depth 2 and 3, with and
without quiescence (4 plies). The script reports, for each player and configuration, the share of
positions where the chosen move is the Stockfish move, the nodes and the seconds per move, and for
depth 3 the paired difference of the agreement (with and without quiescence) and its standard error.

Writes ``data/reports/QUIESCENCE.md``. Nothing is written to the model cards.
"""

import argparse
import math
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import chess

from chess_engine.bots.time_manager import TimeManager
from chess_lab.measure.evaluate import SAMPLE_SEED, TEST, TEST_SAMPLE_ROWS
from chess_lab.players import make_player, validate_specs
from chess_lab.training.sidecar import load_sample

#: (maximum depth, quiescence plies) pairs that are measured.
CONFIGS: Tuple[Tuple[int, int], ...] = ((2, 0), (2, 4), (3, 0), (3, 4))
#: Clock given to the bot, in seconds: large enough for the depth limit to end every search.
GENEROUS_CLOCK = 1e6


def paired_difference(
    with_feature: Sequence[int], without_feature: Sequence[int]
) -> Tuple[float, float]:
    """Mean paired difference of two 0/1 outcome lists and its standard error.

    :param with_feature: Outcomes with the feature, one per position.
    :param without_feature: Outcomes without it, same positions and order.
    :return: ``(mean difference, standard error)``; the error is 0 for fewer than two positions.
    :raises ValueError: If the lists have different lengths.
    """
    if len(with_feature) != len(without_feature):
        raise ValueError("both lists must cover the same positions")
    diffs = [a - b for a, b in zip(with_feature, without_feature, strict=True)]
    n = len(diffs)
    if n < 2:
        return (sum(diffs) / n if n else 0.0), 0.0
    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return mean, math.sqrt(variance / n)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", default="h1,linear_pst_terms", help="comma-separated specs")
    parser.add_argument("--positions", type=int, default=1000)
    parser.add_argument("--raw", type=Path, default=Path("data/lichess/raw"))
    parser.add_argument("--sidecar", type=Path, default=Path("data/lichess/sidecar"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/QUIESCENCE.md"))
    return parser.parse_args(argv)


def measure(
    spec: str, rows: List[Tuple[str, str]], depth: int, quiescence: int
) -> Tuple[List[int], float, float]:
    """Search every position once.

    :param spec: Player specification (see :func:`make_player`).
    :param rows: ``(fen, stockfish move in UCI)`` pairs.
    :param depth: Maximum search depth.
    :param quiescence: Quiescence plies (0 turns it off).
    :return: ``(agreement outcomes, nodes per move, seconds per move)``.
    """
    player = make_player(spec, TimeManager(), max_depth=depth)
    player.quiescence_plies = quiescence
    hits: List[int] = []
    nodes = 0
    seconds = 0.0
    for fen, best in rows:
        started = time.perf_counter()
        move = player.get_move(chess.Board(fen), GENEROUS_CLOCK)
        seconds += time.perf_counter() - started
        hits.append(int(move.uci() == best))
        nodes += player.last_search_stats.nodes_visited
    return hits, nodes / len(rows), seconds / len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison and write the report.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    specs = [s for s in args.players.split(",") if s]
    validate_specs(specs)
    sample = load_sample(args.raw, args.sidecar, TEST, TEST_SAMPLE_ROWS, seed=SAMPLE_SEED)
    rows = [
        (fen, best)
        for fen, best in zip(sample["fen"], sample["best_move"], strict=True)
        if isinstance(best, str)
    ][: args.positions]

    lines = [
        "# Quiescence search at a fixed depth",
        "",
        f"{len(rows)} test positions (seed {SAMPLE_SEED}); agreement is the share of positions where "
        "the chosen move is the Stockfish move.",
        "",
        "| player | depth | quiescence | agreement | nodes/move | s/move |",
        "|---|---|---|---|---|---|",
    ]
    paired = []
    for spec in specs:
        outcomes = {}
        for depth, quiescence in CONFIGS:
            hits, nodes, seconds = measure(spec, rows, depth, quiescence)
            outcomes[(depth, quiescence)] = hits
            lines.append(
                f"| `{spec}` | {depth} | {quiescence} | {sum(hits) / len(hits):.3f} | "
                f"{nodes:.0f} | {seconds:.3f} |"
            )
            print(lines[-1], flush=True)
        mean, error = paired_difference(outcomes[(3, 4)], outcomes[(3, 0)])
        paired.append(f"| `{spec}` | {mean:+.3f} | {error:.3f} |")
    lines += [
        "",
        "Paired difference at depth 3 (quiescence 4 minus none):",
        "",
        "| player | difference | standard error |",
        "|---|---|---|",
        *paired,
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
