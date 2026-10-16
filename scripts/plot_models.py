#!/usr/bin/env python3
"""Plot the model comparison from the model cards into ``docs/img/``.

python scripts/plot_models.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chess_engine.training.cards import PROJECT_ROOT, load_all_cards  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "docs" / "img"


def benchmarked(cards):
    """Cards that have benchmark results.

    :param cards: All model cards.
    :return: The subset with metrics, sorted by name.
    """
    return sorted((c for c in cards if c.get("benchmark")), key=lambda c: c["name"])


def plot_error_vs_time(cards, path: Path) -> None:
    """Scatter plot of mean error against time per evaluation (the accuracy/speed trade-off).

    :param cards: Cards with benchmark results.
    :param path: PNG file to write.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for card in cards:
        bench = card["benchmark"]
        ax.scatter(bench["ms_per_eval"], bench["mae_cp"], s=60)
        ax.annotate(
            card["name"],
            (bench["ms_per_eval"], bench["mae_cp"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Time per evaluation (ms, log scale)")
    ax.set_ylabel("Mean absolute error vs Stockfish (cp)")
    ax.set_title("Accuracy / speed trade-off")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_agreement(cards, path: Path) -> None:
    """Bar chart of agreement with Stockfish and share of predictions within 100 cp.

    :param cards: Cards with benchmark results.
    :param path: PNG file to write.
    """
    names = [c["name"] for c in cards]
    positions = range(len(names))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        [p - 0.2 for p in positions],
        [100 * c["benchmark"]["accuracy"] for c in cards],
        0.4,
        label="Agrees on who is better",
    )
    ax.bar(
        [p + 0.2 for p in positions],
        [100 * c["benchmark"]["within_100"] for c in cards],
        0.4,
        label="Within 100 cp",
    )
    ax.set_xticks(list(positions), names, rotation=20)
    ax.set_ylabel("% of test positions")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title("Agreement with Stockfish")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    """Write both plots for the models that have been benchmarked.

    :return: Process exit code.
    """
    cards = benchmarked(load_all_cards())
    if not cards:
        print(
            "No benchmarked models yet: run python -m chess_engine.training.evaluate first",
            file=sys.stderr,
        )
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_error_vs_time(cards, OUTPUT_DIR / "models_error_vs_time.png")
    plot_agreement(cards, OUTPUT_DIR / "models_agreement.png")
    print(f"Wrote plots to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
