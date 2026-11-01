"""Export a saved linear model to the JSON file the UCI binary embeds (``rust/models/``).

The binary reads plain JSON so that it needs no npz reader. Only linear models are exported.

Example::

    python -m chess_lab.scripts.export_uci_model --model linear_pst_terms
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

from chess_engine.paths import ROOT
from chess_lab.models.definitions import LinearEvalModel, load_model

DEFAULT_OUT = ROOT / "rust" / "models"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options.

    :param argv: Arguments; None reads ``sys.argv``.
    :return: The parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="linear_pst_terms", help="saved model name")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Write the JSON file of the model.

    :param argv: Arguments; None reads ``sys.argv``.
    :return: Exit code.
    """
    args = parse_args(argv)
    model = load_model(args.model)
    if not isinstance(model, LinearEvalModel):
        raise SystemExit(f"{args.model} is a network; only linear models are exported")
    payload = {
        "features": model.feature_set,
        "perspective": model.perspective,
        "weights": [float(w) for w in model.weights],
        "bias": model.bias,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.model}.json"
    target.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(f"{target} ({len(payload['weights'])} weights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
