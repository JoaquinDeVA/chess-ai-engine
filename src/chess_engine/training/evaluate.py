"""Evaluate a model: ``python -m chess_engine.training.evaluate --model <name>``.

By default the model is judged on the fixed test sample of the raw shards (``--raw``,
``--sidecar``): the ``accuracy`` section (:func:`~.judge.judge`) and the ``cost`` section (median
microseconds per single-position evaluation, as the search calls it). Every model can be
evaluated this way, including the older ones that are not retrained.

The older CSV-era models keep their benchmark on a CSV (``--data``), which writes the
``benchmark`` section with the centipawn metrics of :mod:`training.metrics`.

Playing strength is measured by :mod:`training.tournament` instead.
"""

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import chess
import numpy as np

import chess_engine.models  # noqa: F401  (registers the built-in models)
from chess_engine.models.base import CP_SCALE, EvaluationModel
from chess_engine.models.registry import available_models, load_model

from .cards import current_commit, load_card, today, update_card
from .dataset import SPLIT_SEED, TEST, encode_positions, load_evaluations, split_dataset
from .judge import judge
from .metrics import compute_metrics
from .report import update_history_file
from .sidecar import load_sample

PREDICTION_BATCH = 4096
#: Positions of the fixed test sample (the first rows of a seeded sample of the test split).
TEST_SAMPLE_ROWS = 200_000
#: Seed of the test sample; the same seed gives the same positions.
SAMPLE_SEED = 42
#: Leading sample rows used for best-move agreement (each costs one search step).
AGREEMENT_ROWS = 20_000
#: Positions timed for the cost section.
COST_SAMPLES = 1000
DEFAULT_RAW_DIR = Path("data/lichess/raw")
DEFAULT_SIDECAR_DIR = Path("data/lichess/sidecar")


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


def _median_ms(call: Callable[[chess.Board], float], fens: List[str], warmup: int) -> float:
    """Median time of ``call`` over single positions, in milliseconds."""
    boards = [chess.Board(fen) for fen in fens]
    for board in boards[:warmup]:
        call(board)
    timings = []
    for board in boards:
        started = time.perf_counter()
        call(board)
        timings.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(timings)


def measure_latency(model: EvaluationModel, fens: List[str], warmup: int = 20) -> float:
    """Median time to evaluate one position (encoding + inference, batch size 1, CPU).

    This is the cost paid inside a game-tree search, where positions arrive one at a time.

    :param model: Model to time.
    :param fens: Positions to evaluate.
    :param warmup: Calls made before measuring (graph tracing, caches).
    :return: Median milliseconds per evaluation.
    """
    return _median_ms(model.predict_cp, fens, warmup)


def sample_benchmark(
    model: EvaluationModel, raw: Path, sidecar: Path, agreement_n: int = AGREEMENT_ROWS
) -> Dict[str, Any]:
    """Judge a model on the fixed test sample and time its single-position evaluation.

    :param model: Model to evaluate.
    :param raw: Directory with the raw shards.
    :param sidecar: Directory with their sidecars.
    :param agreement_n: Leading sample rows used for best-move agreement.
    :return: ``accuracy`` (judge output plus the sample description) and ``cost``.
    """
    sample = load_sample(raw, sidecar, TEST, TEST_SAMPLE_ROWS, seed=SAMPLE_SEED)
    accuracy = judge(model, sample, agreement_n)
    accuracy["sample"] = {"rows": len(sample), "seed": SAMPLE_SEED, "agreement_n": agreement_n}
    cost_fens = sample["fen"].head(COST_SAMPLES).tolist()
    cost = {
        "us_per_eval": round(_median_ms(model.expected_score, cost_fens, 20) * 1000.0, 2),
        "samples": len(cost_fens),
        "date": today(),
        "commit": current_commit(),
    }
    return {"accuracy": accuracy, "cost": cost}


def benchmark(
    model: EvaluationModel,
    data: Path,
    max_test_rows: Optional[int] = None,
    latency_samples: int = 1000,
    jobs: int = -1,
) -> Dict[str, Any]:
    """Compute the CSV benchmark section of a model card (centipawn metrics).

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


def features_name(model: EvaluationModel) -> str:
    """Name of the feature set of a model, for the card.

    :param model: Model instance.
    :return: ``model.extractor.name`` for the older models, else the class's ``features.name``.
    """
    features = getattr(model, "extractor", None) or getattr(model, "features", None)
    return features.name if features is not None else "—"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--path", type=Path, help="saved model file (default: models/artifacts)")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR, help="raw shards directory")
    parser.add_argument(
        "--sidecar", type=Path, default=DEFAULT_SIDECAR_DIR, help="sidecar directory"
    )
    parser.add_argument(
        "--agreement-n",
        type=int,
        default=AGREEMENT_ROWS,
        help="sample rows used for best-move agreement (lower it for slow models)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="CSV benchmark of the older models instead of the test sample: fen,cp,is_mate,"
        "mate_distance",
    )
    parser.add_argument("--max-test-rows", type=int, help="CSV benchmark: first N test rows")
    parser.add_argument("--latency-samples", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=-1)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Evaluate a model, store the result in its card and refresh ``MODELS_HISTORY.md``.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    model = load_model(args.model, args.path)
    print(f"Evaluating {args.model} ...")

    sections: Dict[str, Any] = {
        "architecture": model.architecture,
        "features": features_name(model),
        "parameters": model.count_parameters(),
    }
    if args.data is None:
        sections.update(sample_benchmark(model, args.raw, args.sidecar, args.agreement_n))
    else:
        sections["benchmark"] = benchmark(
            model, args.data, args.max_test_rows, args.latency_samples, args.jobs
        )
    saved = args.path or model.default_path()
    if saved.exists():
        sections["size_kb"] = round(saved.stat().st_size / 1024, 1)
    if "status" not in load_card(args.model):
        sections["status"] = "current"
    update_card(args.model, **sections)
    update_history_file()

    if args.data is None:
        accuracy, cost = sections["accuracy"], sections["cost"]
        print(f"  sample rows: {accuracy['sample']['rows']:,}")
        print(f"  spearman (all): {accuracy['all']['spearman']:.4f}")
        print(f"  mae (all): {accuracy['all']['mae']:.4f}")
        print(
            f"  best move (all / non-quiet): {accuracy['all']['best_move_agreement']:.4f}"
            f" / {accuracy['non_quiet']['best_move_agreement']:.4f}"
        )
        print(f"  us per eval: {cost['us_per_eval']}")
    else:
        result = sections["benchmark"]
        for key in ("n_test", "mae_cp", "rmse_cp", "accuracy", "within_50", "within_100"):
            print(f"  {key:>12}: {result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
