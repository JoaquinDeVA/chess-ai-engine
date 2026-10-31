"""Evaluate a model: ``python -m chess_lab.measure.evaluate --model <name>``.

By default the model is judged on the fixed test sample of the raw shards (``--raw``,
``--sidecar``): the ``accuracy`` section (:func:`~.judge.judge`) and the ``cost`` section (median
microseconds per single-position evaluation, as the search calls it). Every model can be
evaluated this way, including the older ones that are not retrained.

Playing strength is measured with the UCI engine and fastchess instead.
"""

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import chess
import numpy as np

from chess_engine.models import artifact_path, h1_evaluator, load_h1_weights, split_spec
from chess_lab.measure.judge import judge
from chess_lab.measure.report import update_history_file
from chess_lab.models.cards import current_commit, load_card, today, update_card
from chess_lab.models.defaults import SAMPLE_SEED
from chess_lab.models.definitions import available_models, load_model
from chess_lab.models.python_eval import PythonEvaluator
from chess_lab.models.representation import Representation, score_to_cp
from chess_lab.training.dataset import TEST
from chess_lab.training.sidecar import load_sample

#: Evaluations that are not saved models: the hand-written heuristic.
HEURISTICS = ("h1",)

#: Positions of the fixed test sample (the first rows of a seeded sample of the test split).
TEST_SAMPLE_ROWS = 200_000
#: Leading sample rows used for best-move agreement (each costs one search step).
AGREEMENT_ROWS = 20_000
#: Positions timed for the cost section.
COST_SAMPLES = 1000
DEFAULT_RAW_DIR = Path("data/lichess/raw")
DEFAULT_SIDECAR_DIR = Path("data/lichess/sidecar")


class HeuristicModel:
    """Adapter that lets the heuristic ``h1`` go through the judge.

    It offers what :func:`~.judge.judge` and :func:`sample_benchmark` use of a model:
    ``expected_score`` and ``predict_cp_batch``. The batch method returns the centipawns whose
    default-representation score is exactly the heuristic value, so the judge recovers it.
    """

    architecture = "Hand-crafted heuristic (weights fitted to Stockfish)"

    def __init__(self, name: str) -> None:
        self.name = name
        self.evaluator = h1_evaluator()

    def expected_score(self, board: chess.Board) -> float:
        """Expected score of White in ``[0.01, 0.99]``."""
        return self.evaluator.evaluate(board.fen(), True)

    def predict_cp_batch(self, boards) -> np.ndarray:
        """Centipawns (White's point of view) equivalent to the heuristic scores."""
        scores = np.array([self.expected_score(board) for board in boards], dtype=np.float64)
        return score_to_cp(scores, Representation())

    def count_parameters(self) -> int:
        """Number of weights in ``h1_weights.json``."""
        return len(load_h1_weights())


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
    model: Union[PythonEvaluator, HeuristicModel],
    raw: Path,
    sidecar: Path,
    agreement_n: int = AGREEMENT_ROWS,
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


def features_name(model: Union[PythonEvaluator, HeuristicModel]) -> str:
    """Name of the feature set of a model, for the card.

    :param model: Model instance.
    :return: The class's ``features.name``.
    """
    if isinstance(model, HeuristicModel):
        return model.name
    features = getattr(model, "features", None)
    return features.name if features is not None else "—"


def model_spec_arg(text: str) -> str:
    """Argument type of ``--model``: a registered name, ``name@tag`` or a heuristic (``h1``).

    :param text: The command-line value.
    :return: The value, unchanged.
    :raises argparse.ArgumentTypeError: If the name is not registered or the spec is malformed.
    """
    if text in HEURISTICS:
        return text
    try:
        name, _ = split_spec(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if name not in available_models():
        raise argparse.ArgumentTypeError(
            f"invalid choice: {name!r} (choose from {', '.join([*available_models(), *HEURISTICS])})"
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
    """Evaluate a model, store the result in its card and refresh ``RESULTS.md``.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    heuristic = args.model in HEURISTICS
    model = (
        HeuristicModel(args.model)
        if heuristic
        else PythonEvaluator(load_model(args.model, args.path))
    )
    print(f"Evaluating {args.model} ...")

    sections: Dict[str, Any] = {
        "architecture": model.architecture,
        "features": features_name(model),
        "parameters": model.count_parameters(),
    }
    sections.update(sample_benchmark(model, args.raw, args.sidecar, args.agreement_n))
    if heuristic:
        sections["status"] = "reference"
    else:
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
