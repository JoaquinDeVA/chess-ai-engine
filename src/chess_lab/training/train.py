"""Train a registered model.

Every model is trained from the raw shards and their sidecars, sized by effort:
``python -m chess_lab.training.train --model linear_pst --effort <0-100>``.
"""

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from chess_engine.models import artifact_path
from chess_lab.models.cards import clear_measurements, current_commit, today, update_card
from chess_lab.models.defaults import DEFAULT_MATE_KEEP, SAMPLE_SEED
from chess_lab.models.definitions import available_models, get_model_class
from chess_lab.models.representation import Representation

from .dataset import VALIDATION
from .effort import (
    available_rows,
    calibrate,
    effort_config,
    estimated_seconds,
    fit,
    fit_config,
    search,
)
from .sidecar import load_sample

MAX_VALIDATION_ROWS = 50_000
DEFAULT_RAW_DIR = Path("data/lichess/raw")
DEFAULT_SIDECAR_DIR = Path("data/lichess/sidecar")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument(
        "--output", type=Path, help="where to save (default: src/chess_engine/models/artifacts)"
    )
    parser.add_argument("--epochs", type=int, help="override the configured number of epochs")
    parser.add_argument("--effort", type=int, help="models: effort level 0-100")
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
        "--lr-decay",
        type=float,
        help="multiply the learning rate by this factor on each epoch without validation "
        "improvement; e.g. 0.5",
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
    if args.lr_decay is not None and not 0 < args.lr_decay < 1:
        parser.error(f"--lr-decay must be in (0, 1): {args.lr_decay}")
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
    if args.epochs is not None:
        plan["max_epochs"] = args.epochs
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
    estimate = estimated_seconds(plan, timing, cached=not args.no_cache)
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
        fit_config(plan, best["l2"], best["lr"], args.lr_decay),
        quiet,
        seed=args.seed,
        use_cache=not args.no_cache,
    )

    path = model.save(args.output or artifact_path(model_spec(args)))
    clear_measurements(model_spec(args))
    update_card(
        model_spec(args),
        status="current",
        architecture=model_class.architecture,
        features=model_class.feature_set,
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
    """Train the model chosen on the command line from the raw shards.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    model_class = get_model_class(args.model)

    if args.effort is None:
        print("--effort is required")
        return 2
    try:
        return train_by_effort(args, model_class)
    except (ValueError, TypeError) as error:
        print(error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
