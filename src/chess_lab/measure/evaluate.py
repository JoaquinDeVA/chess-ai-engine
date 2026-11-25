"""Evaluate a model: ``python -m chess_lab.measure.evaluate --model <name>``.

By default the model is judged on the fixed test sample of the raw shards (``--raw``,
``--sidecar``): the ``accuracy`` section (:func:`~.judge.judge`) and the ``cost`` section (median
microseconds per single-position evaluation in the core, from a FEN). Every model can be
evaluated this way, including the older ones that are not retrained.

With ``--games <file>`` (``datasets.build_game_positions``) the model is judged instead on positions
of engine games, the kind its search evaluates: the static metrics of :func:`~.judge.static_metrics`
on all of them and on each quiet slice, saved as the ``games`` section of the card; ``accuracy``
and ``cost`` are left as they were.

Playing strength is measured with the UCI engine and fastchess instead.
"""

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from chess_engine import _core
from chess_lab.datasets.build_move_ranking import OUT as RANKING_TABLE
from chess_lab.datasets.sidecar import load_test_sample
from chess_lab.defaults import RAW_DIR, SAMPLE_SEED, SIDECAR_DIR
from chess_lab.measure.judge import by_quiet, judge, predictions, static_metrics
from chess_lab.measure.ranking import judge_ranking
from chess_lab.models.definitions import available_models, load_model
from chess_lab.models.store.cards import current_commit, load_card, today, update_card
from chess_lab.models.store.loader import artifact_path, split_spec

#: Positions of the fixed test sample (the first rows of a seeded sample of the test split).
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
    sample = load_test_sample(raw, sidecar)
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


def games_benchmark(evaluator: _core.Evaluator, games: Path) -> Dict[str, Any]:
    """Static metrics of an evaluation on the game-position set, overall and per quiet slice.

    :param evaluator: Evaluation to judge, as the engine plays it.
    :param games: Parquet file with ``fen``, ``cp``, ``mate`` and ``quiet``.
    :return: ``{"all", "quiet", "non_quiet"}`` of :func:`~.judge.static_metrics`, plus ``file``.
    """
    table = pd.read_parquet(games)
    pred, truth = predictions(evaluator, table)
    report: Dict[str, Any] = by_quiet(
        lambda mask: static_metrics(pred[mask], truth[mask]), table["quiet"].to_numpy(dtype=bool)
    )
    report["file"] = games.name
    return report


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
    parser.add_argument(
        "--ranking",
        type=Path,
        default=RANKING_TABLE,
        help="move-ranking test set (build_move_ranking); skipped when the file does not exist",
    )
    parser.add_argument(
        "--games",
        type=Path,
        help="judge only on this game-position set (build_game_positions) instead",
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
    if args.games:
        games = games_benchmark(evaluator, args.games)
        update_card(args.model, games=games)
        for name in ("all", "quiet", "non_quiet"):
            part = games[name]
            print(
                f"  games ({name}, {part['n']} positions): spearman {part['spearman']:.4f}, "
                f"mae {part['mae']:.4f}, excess {part['excess']:.4f}"
            )
        return 0
    sections: Dict[str, Any] = {
        "architecture": model.architecture,
        "features": model.feature_set,
        "parameters": model.count_parameters(),
    }
    print(f"Evaluating {args.model} ...")
    sections.update(sample_benchmark(evaluator, args.raw, args.sidecar, args.agreement_n))
    if args.ranking.exists():
        sections["ordering"] = judge_ranking(evaluator, pd.read_parquet(args.ranking))
    saved = args.path or artifact_path(args.model)
    if saved.exists():
        sections["size_kb"] = round(saved.stat().st_size / 1024, 1)
    if "status" not in load_card(args.model):
        sections["status"] = "current"
    update_card(args.model, **sections)

    accuracy, cost = sections["accuracy"], sections["cost"]
    print(f"  sample rows: {accuracy['sample']['rows']:,}")
    print(f"  spearman (all): {accuracy['all']['spearman']:.4f}")
    print(f"  mae (all): {accuracy['all']['mae']:.4f}")
    print(f"  excess (all): {accuracy['all']['excess']:.4f}")
    print(
        f"  best move (all / non-quiet): {accuracy['all']['best_move_agreement']:.4f}"
        f" / {accuracy['non_quiet']['best_move_agreement']:.4f}"
    )
    print(f"  us per eval: {cost['us_per_eval']}")
    if "ordering" in sections:
        for name, part in sections["ordering"].items():
            print(
                f"  ordering ({name}, {part['positions']} positions): tau {part['tau']:.4f}, "
                f"top1 {part['top1']:.4f}, top3 {part['top3']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
