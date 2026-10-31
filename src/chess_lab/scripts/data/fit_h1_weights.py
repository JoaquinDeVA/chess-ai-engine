#!/usr/bin/env python3
"""Fit the weights of the heuristic ``h1`` to Stockfish evaluations.

The six features of ``h1`` are computed for a sample of TRAIN positions without mates. The
weights are the least-squares fit (no intercept) of ``white_score - 0.5`` on ``feature - 0.5``.
The fit is judged on the 20% of the sample selected by the FEN hash, against the plain
``h3`` weights of the older heuristic, and written to the versioned weights file of :func:`h1`.

    python src/chess_lab/scripts/data/fit_h1.py --raw data/lichess/raw --sidecar data/lichess/sidecar \
        --positions 300000 --jobs 4
"""

import argparse
import datetime
import json
import multiprocessing
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import chess
import numpy as np

from chess_engine.evaluation.features import get_position_features
from chess_engine.evaluation.heuristics import (
    FEATURE_NAMES,
    HEURISTIC_MARGIN,
)
from chess_engine.models.representation import Representation, white_score
from chess_engine.models.spec import SAMPLE_SEED
from chess_lab.training.dataset import TRAIN
from chess_lab.training.sidecar import load_sample

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = PROJECT_ROOT / "src" / "chess_engine" / "evaluation" / "h1_weights.json"

#: Held-out R2 of the plain h3 weights (the old fixed-weight heuristic), from the fit of
#: 2026-10-17; h3 no longer exists as a function, so the baseline is stored here.
H3_BASELINE_R2 = 0.0740605

#: Salt of the hash that selects the held-out positions (share :data:`HOLDOUT_SHARE`).
HOLDOUT_SALT = "h1-holdout"
HOLDOUT_SHARE = 0.2
#: Positions per work item of the feature computation.
CHUNK_SIZE = 2000


def features_of(fens: Sequence[str]) -> np.ndarray:
    """Feature matrix of some positions.

    :param fens: FEN strings.
    :return: Array with one row per position and one column per name in ``FEATURE_NAMES``.
    """
    rows = np.empty((len(fens), len(FEATURE_NAMES)))
    for index, fen in enumerate(fens):
        features = get_position_features(chess.Board(fen))
        rows[index] = [features[name] for name in FEATURE_NAMES]
    return rows


def compute_features(fens: Sequence[str], jobs: int) -> np.ndarray:
    """Feature matrix of many positions, computed in ``jobs`` spawned worker processes.

    :param fens: FEN strings.
    :param jobs: Worker processes (1 computes in this process).
    :return: Same layout as :func:`features_of`.
    """
    chunks = [fens[start : start + CHUNK_SIZE] for start in range(0, len(fens), CHUNK_SIZE)]
    if jobs <= 1:
        parts = [features_of(chunk) for chunk in chunks]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
            parts = list(pool.map(features_of, chunks))
    if not parts:
        return np.empty((0, len(FEATURE_NAMES)))
    return np.vstack(parts)


def holdout_mask(fens: Sequence[str]) -> np.ndarray:
    """Positions of the held-out 20%, selected by the hash of their FEN.

    :param fens: FEN strings.
    :return: Boolean array, True for the held-out positions.
    """
    limit = HOLDOUT_SHARE * 10_000
    return np.array(
        [zlib.crc32(f"{HOLDOUT_SALT}:{fen}".encode()) % 10_000 < limit for fen in fens],
        dtype=bool,
    )


def fit_weights(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Least-squares weights of ``targets - 0.5 ~ sum(w * (features - 0.5))``, no intercept.

    :param features: Feature matrix, values in ``[0, 1]`` (0.5 = equal).
    :param targets: White expected scores in ``[0, 1]``.
    :return: One weight per feature column.
    """
    weights, *_ = np.linalg.lstsq(features - 0.5, targets - 0.5, rcond=None)
    return weights


def predict_white(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """White expected score of :func:`h1`: the linear formula clipped to the heuristic margin.

    :param features: Feature matrix.
    :param weights: One weight per feature column.
    :return: Scores in ``[HEURISTIC_MARGIN, 1 - HEURISTIC_MARGIN]``.
    """
    raw = 0.5 + (features - 0.5) @ weights
    return np.clip(raw, HEURISTIC_MARGIN, 1.0 - HEURISTIC_MARGIN)


def r2_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Coefficient of determination of ``predictions`` against ``targets``.

    :param targets: Observed scores.
    :param predictions: Predicted scores.
    :return: ``1 - SSE / SST``; NaN if the targets are constant.
    """
    residual = np.sum((targets - predictions) ** 2)
    total = np.sum((targets - targets.mean()) ** 2)
    return float(1.0 - residual / total) if total > 0 else float("nan")


def sample_positions(raw: Path, sidecar: Path, positions: int) -> Tuple[List[str], np.ndarray]:
    """TRAIN positions without mates, with their White expected scores.

    :param raw: Directory with the raw shards.
    :param sidecar: Directory with their sidecars.
    :param positions: Positions to draw before the mates are dropped.
    :return: ``(fens, targets)``.
    """
    frame = load_sample(raw, sidecar, TRAIN, positions, seed=SAMPLE_SEED)
    frame = frame[frame["mate"].isna()]
    cp = frame["cp"].to_numpy(dtype=float)
    targets = white_score(cp, np.full(len(cp), np.nan), Representation())
    return frame["fen"].tolist(), targets


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/lichess/raw"))
    parser.add_argument("--sidecar", type=Path, default=Path("data/lichess/sidecar"))
    parser.add_argument("--positions", type=int, default=300_000, help="TRAIN positions to draw")
    parser.add_argument("--jobs", type=int, default=4, help="worker processes")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="weights JSON file")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Sample, fit, report both R-squared values and write the weights file.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    fens, targets = sample_positions(args.raw, args.sidecar, args.positions)
    held = holdout_mask(fens)
    fit = ~held
    if int(fit.sum()) < len(FEATURE_NAMES) or not held.any():
        print(f"Too few TRAIN positions without mates ({len(fens)}) to fit and hold out.")
        return 1

    features = compute_features(fens, args.jobs)
    weights = fit_weights(features[fit], targets[fit])
    r2 = r2_score(targets[held], predict_white(features[held], weights))
    r2_h3 = H3_BASELINE_R2

    print(f"positions {len(fens)} (fit {int(fit.sum())}, held out {int(held.sum())})")
    print(f"held-out R2: h1 {r2:.4f}, plain h3 {r2_h3:.4f}, gain {r2 - r2_h3:+.4f}")
    payload = {
        "weights": {name: float(w) for name, w in zip(FEATURE_NAMES, weights, strict=True)},
        "n": len(fens),
        "r2": r2,
        "r2_h3": r2_h3,
        "date": datetime.date.today().isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
