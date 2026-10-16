"""Benchmark a model against Stockfish: ``python -m training.evaluate --model <name> --data <csv>``.

The benchmark uses the fixed hash-based test split of :mod:`training.dataset` and reports
the metrics of :mod:`training.metrics`, the time needed to evaluate one position, and
(optionally) the results of a small match against reference bots.
"""

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chess
import numpy as np

import models  # noqa: F401  (registers the built-in models)
from models.base import CP_SCALE, EvaluationModel
from models.registry import available_models, load_model

from .cards import current_commit, today, update_card
from .dataset import SPLIT_SEED, encode_positions, load_evaluations, split_dataset
from .metrics import compute_metrics
from .report import update_history_file

PREDICTION_BATCH = 4096


def predict_in_batches(
    model: EvaluationModel, inputs, batch_size: int = PREDICTION_BATCH
) -> np.ndarray:
    """Run ``model.predict_target`` over a large encoded dataset in chunks.

    :param model: Model to run.
    :param inputs: Encoded positions (tuple of arrays with the same length).
    :param batch_size: Positions per call.
    :return: Predictions for all positions, shape ``(n,)``.
    """
    size = len(inputs[0])
    parts = [
        model.predict_target(tuple(array[start : start + batch_size] for array in inputs))
        for start in range(0, size, batch_size)
    ]
    return np.concatenate(parts) if parts else np.empty(0)


def measure_latency(model: EvaluationModel, fens: List[str], warmup: int = 20) -> float:
    """Median time to evaluate one position (encoding + inference, batch size 1, CPU).

    This is the cost paid inside a game-tree search, where positions arrive one at a time.

    :param model: Model to time.
    :param fens: Positions to evaluate.
    :param warmup: Calls made before measuring (graph tracing, caches).
    :return: Median milliseconds per evaluation.
    """
    boards = [chess.Board(fen) for fen in fens]
    for board in boards[:warmup]:
        model.predict_cp(board)
    timings = []
    for board in boards:
        started = time.perf_counter()
        model.predict_cp(board)
        timings.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(timings)


def benchmark(
    model: EvaluationModel,
    data: Path,
    max_test_rows: Optional[int] = None,
    latency_samples: int = 1000,
    jobs: int = -1,
) -> Dict[str, Any]:
    """Compute the benchmark section of a model card.

    :param model: Model to evaluate.
    :param data: Evaluations CSV.
    :param max_test_rows: Evaluate only the first N test positions.
    :param latency_samples: Positions used to time single evaluations.
    :param jobs: Processes used to encode positions.
    :return: Metrics plus dataset name, latency, date and code revision.
    """
    frame = load_evaluations(data)
    test = split_dataset(frame)[2]
    if max_test_rows:
        test = test.head(max_test_rows)

    inputs = encode_positions(model.extractor, test["fen"], n_jobs=jobs)
    predicted = np.clip(predict_in_batches(model, inputs), -1.0, 1.0) * CP_SCALE
    truth = test["y"].to_numpy() * CP_SCALE

    result = compute_metrics(predicted, truth)
    sample = test["fen"].sample(min(latency_samples, len(test)), random_state=SPLIT_SEED).tolist()
    result.update(
        dataset=data.name,
        ms_per_eval=round(measure_latency(model, sample), 3),
        date=today(),
        commit=current_commit(),
    )
    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument(
        "--data", required=True, type=Path, help="CSV: fen,cp,is_mate,mate_distance"
    )
    parser.add_argument("--path", type=Path, help="saved model file (default: models/artifacts)")
    parser.add_argument("--max-test-rows", type=int)
    parser.add_argument("--latency-samples", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument(
        "--strength", action="store_true", help="also play a match against reference bots"
    )
    parser.add_argument("--games", type=int, default=10, help="games per opponent for --strength")
    parser.add_argument(
        "--depth", type=int, default=2, help="search depth of the ML bot for --strength"
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Benchmark a model, store the result in its card and refresh ``MODELS_HISTORY.md``.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    model = load_model(args.model, args.path)
    print(f"Benchmarking {args.model} ...")
    result = benchmark(model, args.data, args.max_test_rows, args.latency_samples, args.jobs)

    sections: Dict[str, Any] = {
        "architecture": model.architecture,
        "features": model.extractor.name,
        "parameters": model.count_parameters(),
        "benchmark": result,
    }
    saved = args.path or model.default_path()
    if saved.exists():
        sections["size_kb"] = round(saved.stat().st_size / 1024, 1)
    update_card(args.model, **sections)

    if args.strength:
        from .strength import run_strength_match

        update_card(args.model, strength=run_strength_match(args.model, args.games, args.depth))

    update_history_file()
    for key in (
        "n_test",
        "mae_cp",
        "rmse_cp",
        "accuracy",
        "within_50",
        "within_100",
        "ms_per_eval",
    ):
        print(f"  {key:>12}: {result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
