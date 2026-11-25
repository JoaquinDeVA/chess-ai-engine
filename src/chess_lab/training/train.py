"""Train a registered model.

Every model is trained from the raw shards and their sidecars, sized by effort:
``python -m chess_lab.training.train --model net_pst_threats_w64_both --effort <0-100>``.
With ``--init <model@tag> --tag <tag>`` the fit starts from a trained network instead of random
weights (:func:`~.neural.warm_layers`: the new network starts with its exact output).
"""

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from chess_engine.paths import ROOT
from chess_lab.datasets.sidecar import count_rows, data_fingerprint, load_sample
from chess_lab.datasets.split import TRAIN, VALIDATION
from chess_lab.defaults import (
    RAW_DIR,
    SAMPLE_SEED,
    SIDECAR_DIR,
    format_duration,
)
from chess_lab.models.definitions import available_models, get_model_class, load_model
from chess_lab.models.store.cards import clear_measurements, current_commit, today, update_card
from chess_lab.models.store.loader import artifact_path
from chess_lab.models.target import DEFAULT_MATE_KEEP, Representation

from .effort import (
    calibrate,
    effort_config,
    estimated_seconds,
    fit,
    fit_config,
    search,
)
from .neural import checkpoint_config, environment, warm_layers

MAX_VALIDATION_ROWS = 50_000


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--output", type=Path, help="where to save (default: data/models)")
    parser.add_argument("--effort", type=int, help="models: effort level 0-100")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED, help="seed of the TRAIN sample")
    parser.add_argument(
        "--representation",
        help='"default", or a JSON object with the fields of Representation (scale, k, mate_keep)',
    )
    parser.add_argument(
        "--mate-keep",
        type=float,
        default=DEFAULT_MATE_KEEP,
        help="share of the mate positions kept in the TRAIN and validation sets, in (0, 1] "
        f"(default {DEFAULT_MATE_KEEP}; another value needs --tag)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate of the final fit instead of the one the search picks; e.g. a low "
        "one with --init, so the first epochs do not undo the source",
    )
    parser.add_argument(
        "--tag",
        help="name a variant: artifact and card <model>@e<effort>-<tag> (letters, digits, ._-)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="do not cache the encoded training batches"
    )
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="raw shards directory")
    parser.add_argument("--sidecar", type=Path, default=SIDECAR_DIR, help="sidecar directory")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="file left after every epoch of the final fit; if it exists the fit resumes from it "
        "(without the trials) and it is deleted once the model is saved",
    )
    parser.add_argument(
        "--init",
        help="start from this trained network (name@tag) instead of random weights; needs --tag",
    )
    args = parser.parse_args(argv)
    if args.init is not None and args.tag is None:
        parser.error("--init needs --tag: the warm start would overwrite the untagged model")
    if args.tag is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", args.tag):
        parser.error(f"--tag must contain only letters, digits, '.', '_' or '-': {args.tag!r}")
    if args.lr is not None and not args.lr > 0:
        parser.error(f"--lr must be positive: {args.lr}")
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
    """Name of the model being trained: ``<model>@e<effort>`` or ``<model>@e<effort>-<tag>``.

    :param args: Parsed arguments (``model``, ``effort`` and ``tag``).
    :return: The specification, also used for the artifact and the card.
    """
    variant = f"-{args.tag}" if args.tag else ""
    return f"{args.model}@e{args.effort}{variant}"


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


def train_by_effort(args: argparse.Namespace, model_class: type) -> int:
    """Train a network from the raw shards, sized by ``--effort``.

    Prints the configuration and a duration estimate, then runs the trials and the final fit.
    With an existing ``--checkpoint`` the final fit resumes from it, with its configuration, and
    the timing and the trials are skipped.

    :param args: Parsed arguments (``effort``, ``representation``, ``raw``, ``sidecar``, ...).
    :param model_class: A concrete :class:`NeuralEvalModel` subclass.
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
    initial = None
    if args.init is not None:
        source = load_model(args.init)
        if Representation.from_dict(source.rep) != rep:
            print(f"{args.init} was trained with another representation: {source.rep}")
            return 2
        initial = warm_layers(source, model_class)
        print(f"Warm start from {args.init}")
    available = count_rows(args.raw, args.sidecar, TRAIN, quiet_only=True, mate_keep=rep.mate_keep)
    if available == 0:
        print(f"no TRAIN rows found in {args.sidecar}")
        return 2
    plan = effort_config(args.effort, available)
    data = data_fingerprint(args.raw, args.sidecar)
    recipe = {
        "model": args.model,
        "effort": args.effort,
        "positions": plan["positions"],
        "seed": args.seed,
        "representation": rep.to_dict(),
        "init": args.init,
        "lr": args.lr,
        "data": data,
    }
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
        quiet_only=True,
        mate_keep=rep.mate_keep,
    )
    config = checkpoint_config(args.checkpoint)
    if config is not None:
        expected = fit_config(plan, config["l2"], config["lr"])
        changed = sorted(k for k in expected if k != "workers" and config.get(k) != expected[k])
        left = config.get("recipe", {})
        changed += [f"recipe.{k}" for k in recipe if left.get(k) != recipe[k]]
        if changed:
            print(
                f"{args.checkpoint} was left by a fit with other {', '.join(changed)}: "
                "remove it to start again"
            )
            return 2
        print(f"Resuming the final fit from {args.checkpoint}")
    else:
        timing = calibrate(model_class, rep, args.raw, args.sidecar, validation, plan, True)
        estimate = estimated_seconds(plan, timing, cached=not args.no_cache)
        print(
            f"Timed: {timing.seconds_per_position * 1e6:.1f} us per position, "
            f"validation {timing.encode_seconds:.2f} s to featurize and "
            f"{timing.score_seconds:.2f} s per epoch"
        )
        print(
            f"Estimated duration (upper bound, early stopping ends sooner): {format_duration(estimate)}"
        )
        best = search(model_class, rep, args.raw, args.sidecar, validation, plan, True)
        lr = best["lr"] if args.lr is None else args.lr
        config = fit_config(plan, best["l2"], lr)
        # Kept in the checkpoint, so a resumed run still counts the timing and the trials and
        # refuses to go on under another recipe.
        config["setup_seconds"] = round(time.perf_counter() - started, 3)
        config["recipe"] = recipe
    print(f"Final fit with l2={config['l2']:.2e}, lr={config['lr']:.2e}")
    model, summary = fit(
        model_class,
        rep,
        args.raw,
        args.sidecar,
        validation,
        plan["positions"],
        config,
        True,
        seed=args.seed,
        use_cache=not args.no_cache,
        checkpoint=args.checkpoint,
        initial=initial,
    )

    path = model.save(args.output or artifact_path(model_spec(args)))
    if args.checkpoint is not None:
        args.checkpoint.unlink(missing_ok=True)
    clear_measurements(model_spec(args))
    summary["config"] = {k: v for k, v in summary["config"].items() if k != "recipe"}
    raw = args.raw.resolve()
    update_card(
        model_spec(args),
        status="current",
        architecture=model_class.architecture,
        features=model_class.feature_set,
        parameters=model.count_parameters(),
        size_kb=round(path.stat().st_size / 1024, 1),
        training={
            "dataset": str(raw.relative_to(ROOT) if raw.is_relative_to(ROOT) else raw),
            "data": data,
            "date": today(),
            "commit": current_commit(),
            "environment": environment(),
            "recipe": recipe,
            **summary,
            "seconds": round(config["setup_seconds"] + summary["seconds"], 3),
            "train_samples": summary["train_rows"],
            "effort": args.effort,
            "effort_config": plan,
            "representation": rep.to_dict(),
            **({"init": args.init} if args.init else {}),
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
