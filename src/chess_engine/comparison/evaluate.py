"""Evaluate a model: ``python -m chess_engine.comparison.evaluate --model <name>``.

By default the model is judged on the fixed test sample of the raw shards (``--raw``,
``--sidecar``): the ``accuracy`` section (:func:`~.judge.judge`) and the ``cost`` section (median
microseconds per single-position evaluation, as the search calls it). Every model can be
evaluated this way, including the older ones that are not retrained.

Playing strength is measured by :mod:`comparison.tournament` instead.
"""

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import chess

import chess_engine.models  # noqa: F401  (registers the built-in models)
from chess_engine.models.base import EvaluationModel
from chess_engine.models.cards import current_commit, load_card, today, update_card
from chess_engine.models.registry import artifact_path, available_models, load_model, split_spec
from chess_engine.training.dataset import TEST
from chess_engine.training.sidecar import load_sample
from chess_engine.training.spec import SAMPLE_SEED

from .judge import judge
from .report import update_history_file

#: Positions of the fixed test sample (the first rows of a seeded sample of the test split).
TEST_SAMPLE_ROWS = 200_000
#: Leading sample rows used for best-move agreement (each costs one search step).
AGREEMENT_ROWS = 20_000
#: Positions timed for the cost section.
COST_SAMPLES = 1000
DEFAULT_RAW_DIR = Path("data/lichess/raw")
DEFAULT_SIDECAR_DIR = Path("data/lichess/sidecar")


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


def features_name(model: EvaluationModel) -> str:
    """Name of the feature set of a model, for the card.

    :param model: Model instance.
    :return: ``model.extractor.name`` for the older models, else the class's ``features.name``.
    """
    features = getattr(model, "extractor", None) or getattr(model, "features", None)
    return features.name if features is not None else "—"


def model_spec_arg(text: str) -> str:
    """Argument type of ``--model``: a registered name, or ``name@tag``.

    :param text: The command-line value.
    :return: The value, unchanged.
    :raises argparse.ArgumentTypeError: If the name is not registered or the spec is malformed.
    """
    try:
        name, _ = split_spec(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if name not in available_models():
        raise argparse.ArgumentTypeError(
            f"invalid choice: {name!r} (choose from {', '.join(available_models())})"
        )
    return text


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, type=model_spec_arg, help="model name, or name@tag"
    )
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
    sections.update(sample_benchmark(model, args.raw, args.sidecar, args.agreement_n))
    saved = args.path or artifact_path(args.model)
    if saved.exists():
        sections["size_kb"] = round(saved.stat().st_size / 1024, 1)
    if "status" not in load_card(args.model):
        sections["status"] = "current"
    update_card(args.model, **sections)
    update_history_file()

    accuracy, cost = sections["accuracy"], sections["cost"]
    print(f"  sample rows: {accuracy['sample']['rows']:,}")
    print(f"  spearman (all): {accuracy['all']['spearman']:.4f}")
    print(f"  mae (all): {accuracy['all']['mae']:.4f}")
    print(
        f"  best move (all / non-quiet): {accuracy['all']['best_move_agreement']:.4f}"
        f" / {accuracy['non_quiet']['best_move_agreement']:.4f}"
    )
    print(f"  us per eval: {cost['us_per_eval']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
