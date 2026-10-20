#!/usr/bin/env python3
"""Profile search depth against search accuracy at a fixed time per move.

python scripts/bench/profile_depth_vs_accuracy.py --players linear_pst,h1 --seconds 1,3,10 --positions 300

Use ``--parallel 1`` (the default) for published numbers: parallel workers are limited to one
thread each but still compete for the machine.

Writes the JSON results, ``data/reports/TRADEOFF.md`` and ``docs/img/depth_vs_accuracy.png``.
The ``tradeoff`` section of the model cards of the profiled models (those that have a card) is
updated too.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chess_engine.comparison.tradeoff import record_cards, run_profiles  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PLAYERS = "linear_pst,linear_pst_terms,linear_material,h1,mlp_features"


def render_markdown(results, path: Path) -> None:
    """Write the results table.

    :param results: Output of :func:`run_profiles`.
    :param path: Markdown file to write.
    """
    lines = [
        "# Depth against accuracy at a fixed time per move",
        "",
        "| Player | Seconds | Mean depth | Median depth | Nodes/s | Search agreement "
        "(all / quiet / non-quiet) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        overall = r["overall"]
        if "mean_depth" not in overall:
            continue
        agreement = " / ".join(
            f"{r[key].get('search_agreement', float('nan')):.3f}"
            for key in ("overall", "quiet", "non_quiet")
        )
        lines.append(
            f"| {r['player']} | {r['seconds']:g} | {overall['mean_depth']:.2f} | "
            f"{overall['median_depth']:.1f} | {overall['nodes_per_second']:.0f} | {agreement} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def plot(results, path: Path) -> None:
    """Scatter of completed depth against search agreement, one point per player and time.

    :param results: Output of :func:`run_profiles`.
    :param path: PNG file to write.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in results:
        overall = r["overall"]
        if "mean_depth" not in overall:
            continue
        ax.scatter(overall["mean_depth"], overall["search_agreement"], s=50)
        ax.annotate(
            f"{r['player']} {r['seconds']:g}s",
            (overall["mean_depth"], overall["search_agreement"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_xlabel("Mean completed depth")
    ax.set_ylabel("Search best-move agreement")
    ax.set_title("Depth against accuracy at a fixed time per move")
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Sample positions, profile every player and time, and write the outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", default=DEFAULT_PLAYERS, help="comma-separated specs")
    parser.add_argument("--seconds", default="1,3,10", help="comma-separated seconds per move")
    parser.add_argument("--positions", type=int, default=300)
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="worker processes, each limited to one thread; published numbers should use 1",
    )
    parser.add_argument("--raw", type=Path, default=Path("data/lichess/raw"))
    parser.add_argument("--sidecar", type=Path, default=Path("data/lichess/sidecar"))
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/reports/tradeoff.json")
    parser.add_argument("--markdown", type=Path, default=PROJECT_ROOT / "data/reports/TRADEOFF.md")
    parser.add_argument(
        "--plot", type=Path, default=PROJECT_ROOT / "docs/img/depth_vs_accuracy.png"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    from chess_engine.training.dataset import TEST
    from chess_engine.training.sidecar import load_sample

    positions = load_sample(args.raw, args.sidecar, TEST, args.positions)
    outcome = run_profiles(
        args.players.split(","),
        [float(s) for s in args.seconds.split(",")],
        positions,
        args.parallel,
    )
    results = outcome["results"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(outcome, indent=2))
    render_markdown(results, args.markdown)
    plot(results, args.plot)
    print(f"Wrote {args.out}, {args.markdown} and {args.plot}")
    recorded = record_cards(results)
    if recorded:
        print("Model cards updated: " + ", ".join(recorded))
    if outcome["skipped"]:
        print("SKIPPED (no saved model): " + ", ".join(outcome["skipped"]))


if __name__ == "__main__":
    main()
