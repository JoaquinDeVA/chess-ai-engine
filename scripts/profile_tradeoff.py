#!/usr/bin/env python3
"""Profile search depth against search accuracy at a fixed time per move.

python scripts/profile_tradeoff.py --players pst_tapered,h2,h3 --seconds 1,3,10 --positions 300

Writes the JSON results, ``docs/training/TRADEOFF.md`` and ``docs/img/depth_vs_accuracy.png``.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chess_engine.training.tradeoff import run_profiles  # noqa: E402

DEFAULT_PLAYERS = "pst_tapered,pst_terms,material_linear,h2,h3,mlp_features"


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
    parser.add_argument("--parallel", type=int, default=1, help="worker processes")
    parser.add_argument("--raw", type=Path, default=Path("data/lichess/raw"))
    parser.add_argument("--sidecar", type=Path, default=Path("data/lichess/sidecar"))
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs/training/tradeoff.json")
    parser.add_argument("--markdown", type=Path, default=PROJECT_ROOT / "docs/training/TRADEOFF.md")
    parser.add_argument(
        "--plot", type=Path, default=PROJECT_ROOT / "docs/img/depth_vs_accuracy.png"
    )
    args = parser.parse_args()

    from chess_engine.training.dataset import TEST
    from chess_engine.training.sidecar import load_sample

    positions = load_sample(args.raw, args.sidecar, TEST, args.positions)
    results = run_profiles(
        args.players.split(","),
        [float(s) for s in args.seconds.split(",")],
        positions,
        args.parallel,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    render_markdown(results, args.markdown)
    plot(results, args.plot)
    print(f"Wrote {args.out}, {args.markdown} and {args.plot}")


if __name__ == "__main__":
    main()
