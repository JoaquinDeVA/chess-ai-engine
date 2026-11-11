"""Evaluate a model: ``python -m chess_lab.measure.evaluate --model <name>``.

By default the model is judged on the fixed test sample of the raw shards (``--raw``,
``--sidecar``): the ``accuracy`` section (:func:`~.judge.judge`) and the ``cost`` section (median
microseconds per single-position evaluation in the core, from a FEN). Every model can be
evaluated this way, including the older ones that are not retrained.

Playing strength is measured with the UCI engine and fastchess instead.
"""

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from chess_engine import _core
from chess_lab.datasets.sidecar import load_sample
from chess_lab.datasets.split import TEST
from chess_lab.measure.judge import judge
from chess_lab.measure.report import update_history_file
from chess_lab.models.defaults import RAW_DIR, SAMPLE_SEED, SIDECAR_DIR
from chess_lab.models.definitions import available_models, load_model
from chess_lab.models.store.cards import current_commit, load_card, today, update_card
from chess_lab.models.store.loader import artifact_path, split_spec

#: Positions of the fixed test sample (the first rows of a seeded sample of the test split).
TEST_SAMPLE_ROWS = 200_000
#: Leading sample rows used for best-move agreement (each costs one search step).
AGREEMENT_ROWS = 20_000
#: Positions timed for the cost section.
COST_SAMPLES = 1000


def _median_us(evaluator: _core.Evaluator, fens: List[str], warmup: int) -> float:
    """Median time of one evaluation of White's score, in microseconds."""
    for fen in fens[:warmup]:
        evaluator.evaluate(fen, True)
    timings = []
    for fen in fens:
        started = time.perf_counter()
        evaluator.evaluate(fen, True)
        timings.append((time.perf_counter() - started) * 1e6)
    return statistics.median(timings)


def sample_benchmark(
    evaluator: _core.Evaluator,
    raw: Path,
    sidecar: Path,
    agreement_n: int = AGREEMENT_ROWS,
) -> Dict[str, Any]:
    """Judge an evaluation on the fixed test sample and time it on single positions.

    :param evaluator: Evaluation to judge, as the engine plays it.
    :param raw: Directory with the raw shards.
    :param sidecar: Directory with their sidecars.
    :param agreement_n: Leading sample rows used for best-move agreement.
    :return: ``accuracy`` (judge output plus the sample description) and ``cost``.
    """
    sample = load_sample(raw, sidecar, TEST, TEST_SAMPLE_ROWS, seed=SAMPLE_SEED)
    accuracy = judge(evaluator, sample, agreement_n)
    accuracy["sample"] = {"rows": len(sample), "seed": SAMPLE_SEED, "agreement_n": agreement_n}
    cost_fens = sample["fen"].head(COST_SAMPLES).tolist()
    cost = {
        "us_per_eval": round(_median_us(evaluator, cost_fens, 20), 2),
        "samples": len(cost_fens),
        "date": today(),
        "commit": current_commit(),
    }
    return {"accuracy": accuracy, "cost": cost}


def model_spec_arg(text: str) -> str:
    """Argument type of ``--model``: a registered name or ``name@tag``.

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
    parser.add_argument("--path", type=Path, help="saved model file (default: data/models)")
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="raw shards directory")
    parser.add_argument("--sidecar", type=Path, default=SIDECAR_DIR, help="sidecar directory")
    parser.add_argument(
        "--agreement-n",
        type=int,
        default=AGREEMENT_ROWS,
        help="sample rows used for best-move agreement (lower it for slow models)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Evaluate a model, store the result in its card and refresh ``RESULTS.md``.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    model = load_model(args.model, args.path)
    evaluator = model.evaluator()
    sections: Dict[str, Any] = {
        "architecture": model.architecture,
        "features": model.feature_set,
        "parameters": model.count_parameters(),
    }
    print(f"Evaluating {args.model} ...")
    sections.update(sample_benchmark(evaluator, args.raw, args.sidecar, args.agreement_n))
    saved = args.path or artifact_path(args.model)
    if saved.exists():
        sections["size_kb"] = round(saved.stat().st_size / 1024, 1)
    if "status" not in sections and "status" not in load_card(args.model):
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
