"""Train a registered model: ``python -m training.train --model <name> --data <csv>``."""

import argparse
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional

import models  # noqa: F401  (registers the built-in models)
from models.registry import available_models, get_model_class

from .cards import current_commit, today, update_card
from .dataset import SPLIT_SEED, encode_positions, load_evaluations, split_dataset

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
MAX_VALIDATION_ROWS = 50_000


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
    parser.add_argument(
        "--data", required=True, type=Path, help="CSV: fen,cp,is_mate,mate_distance"
    )
    parser.add_argument(
        "--config", type=Path, help="TOML file (default: training/configs/<model>.toml)"
    )
    parser.add_argument("--output", type=Path, help="where to save (default: models/artifacts)")
    parser.add_argument("--max-rows", type=int, help="read only the first N rows of the CSV")
    parser.add_argument("--train-rows", type=int, help="subsample the training split to N rows")
    parser.add_argument("--epochs", type=int, help="override the configured number of epochs")
    parser.add_argument("--jobs", type=int, default=-1, help="processes used to encode positions")
    return parser.parse_args(argv)


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

    path = model.save(args.output)
    update_card(
        args.model,
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
        f"Saved {path}. Next: python -m training.evaluate --model {args.model} --data {args.data}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
