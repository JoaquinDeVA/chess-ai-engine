"""Loading of the trained evaluations: the app reads a saved model and hands it to the Rust core.

A saved model is a ``.npz`` file in :data:`ARTIFACTS_DIR` written by the laboratory
(``chess_lab.models.definitions``). The core computes the features and the score, so nothing here
evaluates anything: it only turns a file into a ``_core.Evaluator``.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from chess_engine import _core

#: Folder of the trained model files.
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

#: Point of view of the score when the saved representation does not say.
DEFAULT_PERSPECTIVE = "side_to_move"

#: Names of the features of ``h1``, in the order the core takes their weights.
FEATURE_NAMES = (
    "material_diff_normalized",
    "positional_diff_normalized",
    "mobility_diff_normalized",
    "center_diff_normalized",
    "king_safety_diff_normalized",
    "development_diff_normalized",
)

#: Versioned JSON file with the fitted weights of ``h1``.
H1_WEIGHTS_PATH = Path(__file__).with_name("h1_weights.json")


def split_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Split a model specification ``name`` or ``name@tag``.

    A tag names a second artifact of the same model (e.g. trained on undersampled mates).

    :param spec: Model name, optionally followed by ``@`` and a tag.
    :return: ``(name, tag)``; the tag is None for an untagged name.
    :raises ValueError: If the tag after ``@`` or the name before it is empty.
    """
    name, separator, tag = spec.partition("@")
    if not separator:
        return name, None
    if not name or not tag:
        raise ValueError(f"model spec must look like name@tag, got {spec!r}")
    return name, tag


def artifact_path(spec: str) -> Path:
    """Where the artifact of a model specification is saved.

    :param spec: Model name, or ``name@tag``.
    :return: ``ARTIFACTS_DIR/<spec>.npz``.
    """
    split_spec(spec)
    return ARTIFACTS_DIR / f"{spec}.npz"


def installed_models() -> list[str]:
    """Names of the models that have a saved artifact (tagged variants excluded).

    :return: Sorted list of names.
    """
    return sorted(path.stem for path in ARTIFACTS_DIR.glob("*.npz") if "@" not in path.stem)


def load_evaluator(spec: str, path: Optional[Path] = None) -> "_core.Evaluator":
    """The native evaluation of a saved model.

    The file says whether it is a linear model (``weights``, ``bias``) or a sparse network
    (``W0``, ``b0``, ...); the core rejects weights that do not fit the feature set.

    :param spec: Model name, or ``name@tag`` for a tagged artifact.
    :param path: Saved file; defaults to :func:`artifact_path`.
    :return: Evaluator that runs inside the core.
    :raises FileNotFoundError: If the model file does not exist (train it first).
    """
    path = Path(path) if path else artifact_path(spec)
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}")
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        feature_set = meta["features"]
        perspective = meta["rep"].get("perspective", DEFAULT_PERSPECTIVE)
        if "W0" not in data:
            weights = np.ascontiguousarray(data["weights"], dtype=np.float64)
            return _core.Evaluator.linear(feature_set, perspective, weights, float(data["bias"]))
        layers = []
        while f"W{len(layers)}" in data:
            weights, bias = data[f"W{len(layers)}"], data[f"b{len(layers)}"]
            layers.append(
                (
                    np.ascontiguousarray(weights, dtype=np.float64),
                    np.ascontiguousarray(bias, dtype=np.float64),
                )
            )
    return _core.Evaluator.network(feature_set, perspective, layers)


def load_h1_weights() -> dict:
    """The fitted weights of ``h1``.

    :return: Weight per feature name in :data:`FEATURE_NAMES`.
    :raises ValueError: If the file is not keyed by exactly those names.
    """
    weights = json.loads(H1_WEIGHTS_PATH.read_text())["weights"]
    if set(weights) != set(FEATURE_NAMES):
        raise ValueError(f"{H1_WEIGHTS_PATH}: weights must be keyed by {list(FEATURE_NAMES)}")
    return {name: float(weights[name]) for name in FEATURE_NAMES}


def h1_evaluator() -> "_core.Evaluator":
    """The heuristic ``h1`` as a native evaluation.

    :return: Evaluator with the fitted weights of ``h1_weights.json``.
    """
    weights = load_h1_weights()
    return _core.Evaluator.h1([weights[name] for name in FEATURE_NAMES])
