#!/usr/bin/env python3
"""Round-robin at equal time per move: calibrate, play, and write the report.

python scripts/bench/run_tournament.py --players pst_tapered,pst_terms,pst_kb,nn4,nn8,h3t \
    --base 10 --increment 0.1 --workers 5 --hours 10 --out data/tournament/games.jsonl

A short calibration (one opening, every pair) measures the worker-seconds per game; that sizes
the number of openings to fit ``--hours``. The games go to ``--out`` (resumable) and the results
to ``docs/training/TOURNAMENT.md``. The ``tournament`` section of the model cards of the players
(those that have a card) is updated too.
"""

import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from chess_engine.comparison.tournament import (
    ANCHOR,
    paired_openings,
    read_games,
    record_cards,
    run_round_robin,
    size_for_budget,
    summarize,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PLAYERS = "pst_tapered,pst_terms,pst_kb,nn4,nn8,h3t"


def fmt(value: Optional[float], digits: int = 1, signed: bool = False) -> str:
    """Format a number for a table cell; ``—`` when there is none."""
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def render_markdown(
    summary: Dict[str, Any],
    specs: List[str],
    meta: Dict[str, Any],
    anchor: str = ANCHOR,
) -> str:
    """Cross table, scores and Elo of a tournament.

    :param summary: Output of :func:`summarize`.
    :param specs: Players, in display order.
    :param meta: Free-form facts printed above the tables (time control, openings, date).
    :param anchor: Player whose Elo is 0.
    :return: Markdown text.
    """
    players = summary["players"]
    head = summary["head_to_head"]
    lines = [
        "# Round-robin tournament at equal time per move",
        "",
        *(f"- {key}: {value}" for key, value in meta.items()),
        f"- crashed games (excluded): {summary['errors']}",
        "",
        "Scores are from the row player's point of view (wins + half draws). Elo is relative to "
        f"`{anchor}` = 0: *vs {anchor}* uses only the games against it, *fit* is a "
        "Bradley-Terry fit over all pairs. The 95% interval is the normal approximation of the "
        "score. Elo uses scores clipped to [0.01, 0.99]; the intervals treat games as independent, "
        "so they are slightly optimistic (the two games of an opening are correlated). Crashed "
        "games are not retried and are excluded from the scores.",
        "",
        "## Cross table (W-D-L of row against column)",
        "",
        "| Player | " + " | ".join(specs) + " |",
        "|---" * (len(specs) + 1) + "|",
    ]
    for row in specs:
        cells = []
        for col in specs:
            if row == col:
                cells.append("—")
            else:
                wins, draws, losses = head[row][col]
                cells.append(f"{wins}-{draws}-{losses}")
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Scores and Elo",
        "",
        f"| Player | Games | W-D-L | Score % | 95% interval % | Elo vs {anchor} | Elo fit |",
        "|---|---|---|---|---|---|---|",
    ]
    for spec in sorted(specs, key=lambda s: -(players[s]["score"] or 0.0)):
        p = players[spec]
        interval = "—" if p["ci95"] is None else f"{p['ci95'][0]:.1f} – {p['ci95'][1]:.1f}"
        lines.append(
            f"| {spec} | {p['games']} | {p['wins']}-{p['draws']}-{p['losses']} | "
            f"{fmt(p['score'])} | {interval} | {fmt(p['elo_vs_anchor'], 0, True)} | "
            f"{fmt(p['elo'], 0, True)} |"
        )
    return "\n".join(lines) + "\n"


def calibrate(specs: List[str], base: float, increment: float, workers: int) -> Optional[float]:
    """Worker-seconds per game, measured by playing one opening between every pair.

    :return: Wall time times workers over games played, or None if no game was played.
    """
    with tempfile.TemporaryDirectory() as tmp:
        started = time.perf_counter()
        summary = run_round_robin(
            specs,
            paired_openings(1, seed=1),
            base,
            increment,
            workers,
            Path(tmp) / "calibration.jsonl",
        )
        elapsed = time.perf_counter() - started
    if summary["played"] == 0:
        return None
    return elapsed * workers / summary["played"]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", default=DEFAULT_PLAYERS, help="comma-separated specs")
    parser.add_argument("--base", type=float, default=10.0, help="starting clock, seconds")
    parser.add_argument("--increment", type=float, default=0.1, help="seconds per move")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="parallel games; capped at cores - 1",
    )
    parser.add_argument(
        "--anchor", default=ANCHOR, help="player whose Elo is 0; must be one of --players"
    )
    parser.add_argument("--hours", type=float, default=10.0, help="wall-clock budget")
    parser.add_argument("--out", type=Path, default=Path("data/tournament/games.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("docs/training/TOURNAMENT.md"))
    args = parser.parse_args(argv)
    args.specs = [s.strip() for s in args.players.split(",") if s.strip()]
    if len(set(args.specs)) < 2:
        parser.error("--players needs at least two distinct players")
    if args.anchor not in args.specs:
        parser.error(f"--players must include {args.anchor}: it is the Elo anchor")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    """Calibrate, play the tournament, and write the report.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    specs = args.specs
    workers = min(args.workers, max(1, (os.cpu_count() or 2) - 1))
    if workers != args.workers:
        print(f"Using {workers} workers (cores - 1), not {args.workers}.")

    print("Calibrating ...")
    seconds_per_game = calibrate(specs, args.base, args.increment, workers)
    if seconds_per_game is None:
        print("Calibration played no games; check the players.")
        return 1
    openings_per_pair = size_for_budget(args.hours, workers, len(specs), seconds_per_game)
    if openings_per_pair < 1:
        print(f"Budget too small: one opening needs about {seconds_per_game:.1f} s per game.")
        return 1
    print(f"{seconds_per_game:.1f} worker-seconds per game -> {openings_per_pair} openings")

    openings = paired_openings(openings_per_pair)
    run = run_round_robin(specs, openings, args.base, args.increment, workers, args.out)
    print(f"played {run['played']}, skipped {run['skipped']}, crashed {run['errors']}")

    summary = summarize(read_games(args.out), args.anchor)
    meta = {
        "time control": f"{args.base:g}+{args.increment:g} s",
        "workers": workers,
        "openings": openings_per_pair,
        "games per pair": 2 * openings_per_pair,
        "seconds per game (calibration)": f"{seconds_per_game:.1f}",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(summary, specs, meta, args.anchor))
    print(f"Wrote {args.report}")
    recorded = record_cards(summary, specs, args.anchor, args.base, args.increment)
    if recorded:
        print("Model cards updated: " + ", ".join(recorded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
