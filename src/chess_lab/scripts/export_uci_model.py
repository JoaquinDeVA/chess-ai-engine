"""Export a saved network to the JSON file the UCI binary embeds (``rust/models/``).

The binary reads plain JSON so that it needs no npz reader. It embeds ``net_pst_w32`` and loads
any other file with ``setoption name Model value <file>.json``.

Example::

    python -m chess_lab.scripts.export_uci_model --model net_pst_w32@e30
    python -m chess_lab.scripts.export_uci_model --model net_hybrid_w64 --out /tmp/models
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from chess_engine.paths import ROOT
from chess_lab.models.definitions import NeuralEvalModel, load_model

DEFAULT_OUT = ROOT / "rust" / "models"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options.

    :param argv: Arguments; None reads ``sys.argv``.
    :return: The parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="net_pst_w32@e30", help="saved model name")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    return parser.parse_args(argv)


def export_payload(model: NeuralEvalModel) -> Dict[str, Any]:
    """The JSON content of a network.

    :param model: The network.
    :return: Feature set and ``layers``, each with row-major ``weights``.
    """
    layers = [
        {
            "fan_in": int(w.shape[0]),
            "fan_out": int(w.shape[1]),
            "weights": [float(x) for x in w.ravel()],
            "bias": [float(x) for x in b],
        }
        for w, b in model.layers
    ]
    return {"features": model.feature_set, "layers": layers}


def main(argv: Optional[List[str]] = None) -> int:
    """Write the JSON file of the model.

    :param argv: Arguments; None reads ``sys.argv``.
    :return: Exit code.
    """
    args = parse_args(argv)
    model = load_model(args.model)
    payload = export_payload(model)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.model.split('@')[0]}.json"
    target.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(f"{target} ({model.count_parameters()} parameters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
