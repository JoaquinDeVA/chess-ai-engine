#!/usr/bin/env python3
"""Check the train/validation/test split for leakage.

python src/chess_lab/scripts/data/check_split_leakage.py --raw data/lichess/raw --sidecar data/lichess/sidecar

The split assigns each position to train, validation or test by a hash of its FEN. By construction a
position cannot be in two splits, but two *different* positions that are almost the same can: the data
has no game identifiers, so a split by game is not possible. The script measures:

1. duplicates: FENs that appear more than once in the shards, and whether any of them is in two splits;
2. mirror images: test positions whose colour-swapped image (``Board.mirror``) is in the data, and in train;
3. neighbours: test positions with a position one legal move away in the data, and in train.

Neighbours are counted through the moves of the test position (its children); the positions that lead
to it are not searched, so the shares are lower bounds.

With ``--models`` the script also measures whether the neighbours flatter the models: it judges each
model on the fixed test sample of ``comparison.evaluate`` twice, on the positions that have a child in
train and on those that have none, and prints the Spearman correlation and the error of both groups. If
neighbours inflated the accuracy, the first group would score clearly higher.

Writes ``data/reports/SPLIT_LEAKAGE.md``.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import chess
import numpy as np
import pyarrow.parquet as pq

from chess_lab.training.dataset import TEST, TRAIN, VALIDATION

SPLIT_NAMES = {TRAIN: "train", VALIDATION: "validation", TEST: "test"}


def position_key(board: chess.Board) -> str:
    """First four FEN fields: the key of the raw shards."""
    return " ".join(board.fen().split()[:4])


def load_index(raw_dir: Path, sidecar_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Hash and split of every row of the shards.

    :param raw_dir: Folder with the ``part-*.parquet`` shards.
    :param sidecar_dir: Folder with their ``part-*.sidecar.parquet`` files.
    :return: ``(hashes, splits)`` sorted by hash; the hash is Python's string hash, valid only in
        this process.
    """
    hashes, splits = [], []
    for shard in sorted(raw_dir.glob("part-*.parquet")):
        side = pq.read_table(sidecar_dir / f"{shard.stem}.sidecar.parquet", columns=["split"])
        fens = pq.read_table(shard, columns=["fen"])["fen"].to_pylist()
        hashes.append(np.fromiter((hash(f) for f in fens), np.int64, count=len(fens)))
        splits.append(side["split"].to_numpy().astype(np.int8))
    all_hashes, all_splits = np.concatenate(hashes), np.concatenate(splits)
    order = np.argsort(all_hashes, kind="stable")
    return all_hashes[order], all_splits[order]


def duplicate_report(hashes: np.ndarray, splits: np.ndarray) -> Dict[str, int]:
    """Count repeated hashes and the repeated ones that sit in different splits.

    :param hashes: Sorted hashes.
    :param splits: Split of each row, in the same order.
    :return: ``rows``, ``repeated_rows`` (rows whose hash appears again) and ``cross_split_rows``.
    """
    same = hashes[1:] == hashes[:-1]
    repeated = np.zeros(len(hashes), dtype=bool)
    repeated[1:] |= same
    repeated[:-1] |= same
    crossing = np.zeros(len(hashes), dtype=bool)
    differs = same & (splits[1:] != splits[:-1])
    crossing[1:] |= differs
    crossing[:-1] |= differs
    return {
        "rows": int(len(hashes)),
        "repeated_rows": int(repeated.sum()),
        "cross_split_rows": int(crossing.sum()),
    }


def splits_of(hashes: np.ndarray, splits: np.ndarray, keys: Sequence[str]) -> List[int]:
    """Split of each key that is in the index, or -1.

    :param hashes: Sorted hashes of the index.
    :param splits: Splits in the same order.
    :param keys: Position keys to look up.
    :return: One value per key.
    """
    wanted = np.fromiter((hash(k) for k in keys), np.int64, count=len(keys))
    positions = np.searchsorted(hashes, wanted)
    positions[positions == len(hashes)] = len(hashes) - 1
    found = hashes[positions] == wanted
    return np.where(found, splits[positions], -1).tolist()


def neighbour_report(
    hashes: np.ndarray, splits: np.ndarray, test_fens: Sequence[str]
) -> Dict[str, float]:
    """Mirror images and children of test positions that are in the data, and in train.

    :param hashes: Sorted hashes of the index.
    :param splits: Splits in the same order.
    :param test_fens: Positions of the test split to examine.
    :return: ``positions``, and the shares with a mirror image, a mirror in train, a child and a
        child in train.
    """
    mirror_any = mirror_train = child_any = child_train = 0
    for fen in test_fens:
        board = chess.Board(fen)
        mirror = splits_of(hashes, splits, [position_key(board.mirror())])[0]
        mirror_any += mirror != -1
        mirror_train += mirror == TRAIN
        keys = []
        for move in list(board.legal_moves):
            board.push(move)
            keys.append(position_key(board))
            board.pop()
        found = splits_of(hashes, splits, keys) if keys else []
        child_any += any(s != -1 for s in found)
        child_train += any(s == TRAIN for s in found)
    n = max(len(test_fens), 1)
    return {
        "positions": len(test_fens),
        "mirror_in_data": mirror_any / n,
        "mirror_in_train": mirror_train / n,
        "child_in_data": child_any / n,
        "child_in_train": child_train / n,
    }


def has_train_child(hashes: np.ndarray, splits: np.ndarray, fens: Sequence[str]) -> np.ndarray:
    """Whether each position has a position one legal move away that is in train.

    :param hashes: Sorted hashes of the index.
    :param splits: Splits in the same order.
    :param fens: Positions to examine.
    :return: Boolean array, one per position.
    """
    flags = np.zeros(len(fens), dtype=bool)
    for row, fen in enumerate(fens):
        board = chess.Board(fen)
        keys = []
        for move in list(board.legal_moves):
            board.push(move)
            keys.append(position_key(board))
            board.pop()
        if keys:
            flags[row] = any(s == TRAIN for s in splits_of(hashes, splits, keys))
    return flags


def stratified_accuracy(
    model_names: Sequence[str], sample, flags: np.ndarray
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Static accuracy of each model on the positions with and without a neighbour in train.

    :param model_names: Registered model names (``name@tag`` allowed).
    :param sample: Test sample (``fen``, ``cp``, ``mate``, ``best_move``, ``quiet``).
    :param flags: Which rows of the sample have a child in train.
    :return: ``{model: {"with neighbour": metrics, "without neighbour": metrics}}``.
    """
    from chess_engine.models import load_model
    from chess_lab.comparison.judge import judge
    from chess_lab.models.python_eval import PythonEvaluator

    result = {}
    for name in model_names:
        model = PythonEvaluator(load_model(name))
        groups = {}
        for label, mask in (("with neighbour", flags), ("without neighbour", ~flags)):
            part = sample[mask].reset_index(drop=True)
            groups[label] = {**judge(model, part, agreement_n=0)["all"], "rows": len(part)}
        result[name] = groups
    return result


def sample_test_fens(raw_dir: Path, sidecar_dir: Path, count: int, seed: int) -> List[str]:
    """Random test positions, spread over the shards.

    :param raw_dir: Folder with the shards.
    :param sidecar_dir: Folder with the sidecars.
    :param count: How many positions.
    :param seed: Seed of the random choice.
    :return: FEN strings of test rows.
    """
    rng = np.random.default_rng(seed)
    shards = sorted(raw_dir.glob("part-*.parquet"))
    per_shard = max(1, count // len(shards))
    fens: List[str] = []
    for shard in shards:
        split = pq.read_table(sidecar_dir / f"{shard.stem}.sidecar.parquet", columns=["split"])
        rows = np.flatnonzero(split["split"].to_numpy() == TEST)
        if len(rows) == 0:
            continue
        chosen = rng.choice(rows, size=min(per_shard, len(rows)), replace=False)
        column = pq.read_table(shard, columns=["fen"])["fen"].to_pylist()
        fens.extend(column[i] for i in chosen)
    return fens[:count]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/lichess/raw"))
    parser.add_argument("--sidecar", type=Path, default=Path("data/lichess/sidecar"))
    parser.add_argument("--positions", type=int, default=20_000, help="test positions examined")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models", help="comma-separated models to judge on both groups of positions"
    )
    parser.add_argument("--sample-rows", type=int, default=200_000, help="size of that test sample")
    parser.add_argument("--report", type=Path, default=Path("data/reports/SPLIT_LEAKAGE.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks and write the report.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    hashes, splits = load_index(args.raw, args.sidecar)
    duplicates = duplicate_report(hashes, splits)
    counts = {SPLIT_NAMES[s]: int((splits == s).sum()) for s in SPLIT_NAMES}
    test_fens = sample_test_fens(args.raw, args.sidecar, args.positions, args.seed)
    near = neighbour_report(hashes, splits, test_fens)
    lines = [
        "# Split leakage check",
        "",
        f"{duplicates['rows']:,} rows ({', '.join(f'{k} {v:,}' for k, v in counts.items())}).",
        "",
        "## Exact duplicates",
        "",
        f"- Rows whose position appears more than once: {duplicates['repeated_rows']:,}",
        f"- Of those, in two different splits: {duplicates['cross_split_rows']:,}",
        "",
        f"## Near duplicates of {near['positions']:,} test positions (seed {args.seed})",
        "",
        "| relation | in the data | in train |",
        "|---|---|---|",
        f"| colour-swapped image | {near['mirror_in_data']:.1%} | {near['mirror_in_train']:.1%} |",
        f"| a position one move away | {near['child_in_data']:.1%} | {near['child_in_train']:.1%} |",
    ]
    if args.models:
        from chess_lab.comparison.evaluate import SAMPLE_SEED
        from chess_lab.training.sidecar import load_sample

        sample = load_sample(args.raw, args.sidecar, TEST, args.sample_rows, seed=SAMPLE_SEED)
        flags = has_train_child(hashes, splits, sample["fen"].tolist())
        report = stratified_accuracy([m for m in args.models.split(",") if m], sample, flags)
        lines += [
            "",
            f"## Accuracy with and without a neighbour in train ({len(sample):,} test positions, "
            f"{flags.mean():.1%} with a neighbour)",
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
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
