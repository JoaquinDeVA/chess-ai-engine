#!/usr/bin/env python3
"""Measure whether near-duplicates in train flatter the models.

python -m chess_lab.measure.split_leakage --models net_pst_w128_both,net_pst_w64_both

Judges each model on the fixed test sample of ``measure.evaluate`` twice, on the positions that have
a child in train and on those that have none, and prints the Spearman correlation and the error of
both groups. If neighbours inflated the accuracy, the first group would score clearly higher.

The counts of duplicates and neighbours themselves come from ``datasets.check_split_leakage``.
Writes ``data/results/reports/SPLIT_LEAKAGE_MODELS.md``.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from chess_engine.paths import ROOT
from chess_lab.datasets.check_split_leakage import has_train_child, load_index
from chess_lab.datasets.sidecar import load_sample
from chess_lab.datasets.split import TEST
from chess_lab.defaults import RAW_DIR, SAMPLE_SEED, SIDECAR_DIR
from chess_lab.measure.judge import judge
from chess_lab.models.definitions import load_evaluator


def stratified_accuracy(
    model_names: Sequence[str], sample, flags: np.ndarray
) -> dict[str, dict[str, dict[str, float]]]:
    """Static accuracy of each model on the positions with and without a neighbour in train.

    :param model_names: Registered model names (``name@tag`` allowed).
    :param sample: Test sample (``fen``, ``cp``, ``mate``, ``best_move``, ``quiet``).
    :param flags: Which rows of the sample have a child in train.
    :return: ``{model: {"with neighbour": metrics, "without neighbour": metrics}}``.
    """
    result = {}
    for name in model_names:
        evaluator = load_evaluator(name)
        groups = {}
        for label, mask in (("with neighbour", flags), ("without neighbour", ~flags)):
            part = sample[mask].reset_index(drop=True)
            groups[label] = {**judge(evaluator, part, agreement_n=0)["all"], "rows": len(part)}
        result[name] = groups
    return result


def report_lines(report: dict, rows: int, share: float) -> list[str]:
    """The markdown table of :func:`stratified_accuracy`.

    :param report: Its result.
    :param rows: Test positions judged.
    :param share: Share of them with a child in train.
    :return: The lines of the report.
    """
    lines = [
        f"## Accuracy with and without a neighbour in train ({rows:,} test positions, "
        f"{share:.1%} with a neighbour)",
        "",
        "| model | Spearman, with | Spearman, without | MAE, with | MAE, without |",
        "|---|---|---|---|---|",
    ]
    for name, groups in report.items():
        w, wo = groups["with neighbour"], groups["without neighbour"]
        lines.append(
            f"| `{name}` | {w['spearman']:.3f} | {wo['spearman']:.3f} | "
            f"{w['mae']:.4f} | {wo['mae']:.4f} |"
        )
    return lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--sidecar", type=Path, default=SIDECAR_DIR)
    parser.add_argument("--models", required=True, help="comma-separated models to judge")
    parser.add_argument("--sample-rows", type=int, default=200_000, help="size of the test sample")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "data/results/reports/SPLIT_LEAKAGE_MODELS.md"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Judge the models on both groups and write the report.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    hashes, splits = load_index(args.raw, args.sidecar)
    sample = load_sample(args.raw, args.sidecar, TEST, args.sample_rows, seed=SAMPLE_SEED)
    flags = has_train_child(hashes, splits, sample["fen"].tolist())
    report = stratified_accuracy([m for m in args.models.split(",") if m], sample, flags)
    lines = [
        "# Split leakage: accuracy by neighbour",
        "",
        *report_lines(report, len(sample), flags.mean()),
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
