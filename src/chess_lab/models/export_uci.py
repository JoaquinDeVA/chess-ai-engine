"""Export a saved network to the JSON file the UCI binary embeds (``rust/models/``).

The binary reads plain JSON so that it needs no npz reader. It embeds ``net_pst_threats_w128_ob8_both@e100-lr1e-3`` and loads
any other file with ``setoption name Model value <file>.json``.

Example::

    python -m chess_lab.models.export_uci --model net_pst_threats_w128_ob8_both@e100-lr1e-3
    python -m chess_lab.models.export_uci --model net_pst_w64_both@e30 --out /tmp/models
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from chess_engine import _core
from chess_engine.paths import ROOT
from chess_lab.atomic import atomic_path
from chess_lab.models.definitions import NeuralEvalModel, load_model
from chess_lab.models.store.loader import artifact_path

DEFAULT_OUT = ROOT / "rust" / "models"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options.

    :param argv: Arguments; None reads ``sys.argv``.
    :return: The parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", default="net_pst_threats_w128_ob8_both@e100-lr1e-3", help="saved model name"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    return parser.parse_args(argv)


def export_payload(model: NeuralEvalModel, name: str, source_sha256: str) -> Dict[str, Any]:
    """The JSON content of a network.

    :param model: The network.
    :param name: Name of the saved model it comes from (the core does not read it).
    :param source_sha256: SHA-256 of the ``.npz`` it comes from (the core does not read it).
    :return: The ``schema`` the core reads, the provenance, the feature set and ``layers``, each
        with row-major ``weights``.
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
    payload: Dict[str, Any] = {
        "schema": _core.MODEL_SCHEMA,
        "name": name,
        "source_sha256": source_sha256,
        "features": model.feature_set,
        "layers": layers,
        "perspective": "both",
    }
    if model.families:
        payload["families"] = list(model.families)
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    """Write the JSON file of the model.

    :param argv: Arguments; None reads ``sys.argv``.
    :return: Exit code.
    """
    args = parse_args(argv)
    model = load_model(args.model)
    source = artifact_path(args.model)
    payload = export_payload(model, args.model, hashlib.sha256(source.read_bytes()).hexdigest())
    text = json.dumps(payload, allow_nan=False) + "\n"
    # the core must read the file before it is written, so that a bad network leaves no file
    _core.Evaluator.from_json(text)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.model}.json"
    with atomic_path(target) as temporary:
        temporary.write_text(text, encoding="utf-8")
    print(f"{target} ({model.count_parameters()} parameters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
