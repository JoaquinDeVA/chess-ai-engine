"""Train a registered model.

Models of the CSV family: ``python -m chess_engine.training.train --model <name> --data <csv>``.
Linear models, from the raw shards: ``--model linear_pst --effort <0-100>``.
"""

import argparse
import json
import re
import time
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import chess_engine.models  # noqa: F401  (registers the built-in models)
from chess_engine.models.cards import current_commit, today, update_card
from chess_engine.models.linear_eval import LinearEvalModel
from chess_engine.models.registry import artifact_path, available_models, get_model_class

from .dataset import SPLIT_SEED, VALIDATION, encode_positions, load_evaluations, split_dataset
from .effort import (
    SAMPLE_SEED,
    available_rows,
    calibrate,
    effort_config,
    estimated_seconds,
    fit,
    fit_config,
    search,
)
from .representation import DEFAULT_MATE_KEEP, Representation
from .sidecar import load_sample

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
MAX_VALIDATION_ROWS = 50_000
DEFAULT_RAW_DIR = Path("data/lichess/raw")
DEFAULT_SIDECAR_DIR = Path("data/lichess/sidecar")


def load_config(model_name: str, explicit_path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the TOML hyper-parameters of a model.

    :param model_name: Registered model name; ``configs/<name>.toml`` is used by default.
    :param explicit_path: Use this file instead.
    :return: The configuration (empty if there is no file).
    """
    path = explicit_path or CONFIG_DIR / f"{model_name}.toml"
    if not path.exists():
        return {}
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--data", type=Path, help="CSV: fen,cp,is_mate,mate_distance")
    parser.add_argument(
        "--config", type=Path, help="TOML file (default: training/configs/<model>.toml)"
    )
    parser.add_argument("--output", type=Path, help="where to save (default: models/artifacts)")
    parser.add_argument("--max-rows", type=int, help="read only the first N rows of the CSV")
    parser.add_argument("--train-rows", type=int, help="subsample the training split to N rows")
    parser.add_argument("--epochs", type=int, help="override the configured number of epochs")
    parser.add_argument("--jobs", type=int, default=-1, help="processes used to encode positions")
    parser.add_argument("--effort", type=int, help="linear models: effort level 0-100")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED, help="seed of the TRAIN sample")
    parser.add_argument(
        "--representation",
        help='linear models: "default", or a JSON object with the fields of Representation '
        "(scale, k, perspective, filter)",
    )
    parser.add_argument(
        "--mate-keep",
        type=float,
        default=DEFAULT_MATE_KEEP,
        help="share of the mate positions kept in the TRAIN and validation sets, in (0, 1] "
        f"(default {DEFAULT_MATE_KEEP}; another value needs --tag)",
    )
    parser.add_argument(
        "--tag",
        help="train a variant: artifact <model>@<tag>, card <model>@<tag> (letters, digits, ._-)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="do not cache the encoded training batches"
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR, help="raw shards directory")
    parser.add_argument(
        "--sidecar", type=Path, default=DEFAULT_SIDECAR_DIR, help="sidecar directory"
    )
    args = parser.parse_args(argv)
    if args.tag is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", args.tag):
        parser.error(f"--tag must contain only letters, digits, '.', '_' or '-': {args.tag!r}")
    return args


def parse_representation(text: Optional[str]) -> Representation:
    """Read the ``--representation`` option.

    :param text: ``None`` or ``"default"`` for the default representation, otherwise a JSON object
        with the fields of :class:`Representation`.
    :return: The representation.
    :raises ValueError: If the JSON is invalid or a value is not allowed.
    :raises TypeError: If the JSON has a field that Representation does not have.
    """
    if text is None or text == "default":
        return Representation()
    return Representation.from_dict(json.loads(text))


def model_spec(args: argparse.Namespace) -> str:
    """Name of the model being trained: ``<model>`` or ``<model>@<tag>``.

    :param args: Parsed arguments (``model`` and ``tag``).
    :return: The specification, also used for the artifact and the card.
    """
    return f"{args.model}@{args.tag}" if args.tag else args.model


def effort_representation(args: argparse.Namespace) -> Representation:
    """Representation of an effort run: ``--representation`` with ``--mate-keep`` applied.

    :param args: Parsed arguments.
    :return: The representation to train with.
    :raises ValueError: If ``--mate-keep`` disagrees with the ``mate_keep`` of
        ``--representation``, or is outside ``(0, 1]``.
    """
    rep = parse_representation(args.representation)
    if args.mate_keep == DEFAULT_MATE_KEEP:
        return rep
    if rep.mate_keep not in (DEFAULT_MATE_KEEP, args.mate_keep):
        raise ValueError(
            f"--mate-keep {args.mate_keep} disagrees with mate_keep {rep.mate_keep} "
            "of --representation"
        )
    return replace(rep, mate_keep=args.mate_keep)


def _format_duration(seconds: float) -> str:
    """Human-readable duration.

    :param seconds: Duration in seconds.
    :return: Seconds, minutes or hours, with one decimal.
    """
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def train_by_effort(args: argparse.Namespace, model_class: type) -> int:
    """Train a linear model from the raw shards, sized by ``--effort``.

    Prints the configuration and a duration estimate, then runs the trials and the final fit.

    :param args: Parsed arguments (``effort``, ``representation``, ``raw``, ``sidecar``, ...).
    :param model_class: A concrete :class:`LinearEvalModel` subclass.
    :return: Process exit code.
    """
    started = time.perf_counter()
    rep = effort_representation(args)
    if rep.mate_keep != DEFAULT_MATE_KEEP and not args.tag:
        print(
            f"a model with mate_keep other than {DEFAULT_MATE_KEEP} needs --tag (for example "
            "--mate-keep 1.0 --tag mates100): without it the untagged model would be overwritten"
        )
        return 2
    quiet = rep.filter == "quiet"
    available = available_rows(
        args.sidecar, quiet_only=quiet, raw_dir=args.raw, mate_keep=rep.mate_keep
    )
    if available == 0:
        print(f"no TRAIN rows found in {args.sidecar}")
        return 2
    plan = effort_config(args.effort, available)
    print(f"Effort {args.effort}: {available:,} TRAIN rows available")
    for key, value in plan.items():
        print(f"  {key}: {value:,}")
    print(f"  representation: {rep.name}")

    validation = load_sample(
        args.raw,
        args.sidecar,
        VALIDATION,
        MAX_VALIDATION_ROWS,
        seed=SAMPLE_SEED,
        quiet_only=quiet,
        mate_keep=rep.mate_keep,
    )
    timing = calibrate(model_class, rep, args.raw, args.sidecar, validation, plan, quiet)
    estimate = estimated_seconds(plan, timing)
    print(
        f"Timed: {timing.seconds_per_position * 1e6:.1f} us per position, "
        f"validation {timing.encode_seconds:.2f} s to featurize and "
        f"{timing.score_seconds:.2f} s per epoch"
    )
    print(f"Estimated duration: {_format_duration(estimate)}")

    best = search(model_class, rep, args.raw, args.sidecar, validation, plan, quiet)
    print(f"Final fit with l2={best['l2']:.2e}, lr={best['lr']:.2e}")
    model, summary = fit(
        model_class,
        rep,
        args.raw,
        args.sidecar,
        validation,
        plan["positions"],
        fit_config(plan, best["l2"], best["lr"]),
        quiet,
        seed=args.seed,
        use_cache=not args.no_cache,
    )

    path = model.save(args.output or artifact_path(model_spec(args)))
    update_card(
        model_spec(args),
        status="current",
        architecture=model_class.architecture,
        features=model_class.features.name,
        parameters=model.count_parameters(),
        size_kb=round(path.stat().st_size / 1024, 1),
        training={
            "dataset": str(args.raw),
            "date": today(),
            "commit": current_commit(),
            **summary,
            "seconds": round(time.perf_counter() - started, 3),
            "train_samples": summary["train_rows"],
            "effort": args.effort,
            "effort_config": plan,
            "representation": rep.to_dict(),
        },
    )
    print(f"Saved {path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Run the training pipeline: load, split, encode, fit, save, update the model card.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    model_class = get_model_class(args.model)
    if not model_class.trainable:
        print(f"{args.model} is a frozen artifact and cannot be retrained")
        return 2

    is_linear = issubclass(model_class, LinearEvalModel)
    if is_linear != (args.effort is not None):
        print("--effort is required for linear models and only applies to them")
        return 2
    if is_linear:
        try:
            return train_by_effort(args, model_class)
        except (ValueError, TypeError) as error:
            print(error)
            return 2
    if args.data is None:
        print("--data is required for CSV models")
        return 2
    if args.mate_keep != DEFAULT_MATE_KEEP:
        print("--mate-keep only applies to linear models trained with --effort")
        return 2

    config = load_config(args.model, args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs

    print(f"Loading {args.data} ...")
    frame = load_evaluations(args.data, max_rows=args.max_rows)
    train, validation, _ = split_dataset(frame)
    if args.train_rows and args.train_rows < len(train):
        train = train.sample(args.train_rows, random_state=SPLIT_SEED)
    validation = validation.head(MAX_VALIDATION_ROWS)
    print(f"{len(train):,} train / {len(validation):,} validation positions")

    extractor = model_class.extractor
    train_inputs = encode_positions(extractor, train["fen"], n_jobs=args.jobs)
    validation_inputs = encode_positions(extractor, validation["fen"], n_jobs=args.jobs)

    model = model_class.new(config)
    summary = model.fit(
        train_inputs,
        train["y"].to_numpy(),
        validation=(validation_inputs, validation["y"].to_numpy()),
        config=config,
    )

    path = model.save(args.output or artifact_path(model_spec(args)))
    update_card(
        model_spec(args),
        status="current",
        architecture=model_class.architecture,
        features=extractor.name,
        parameters=model.count_parameters(),
        size_kb=round(path.stat().st_size / 1024, 1),
        training={
            "dataset": args.data.name,
            "date": today(),
            "commit": current_commit(),
            "config": config,
            **summary,
        },
    )
    print(
        f"Saved {path}. Next: python -m chess_engine.comparison.evaluate --model {args.model} --data {args.data}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
