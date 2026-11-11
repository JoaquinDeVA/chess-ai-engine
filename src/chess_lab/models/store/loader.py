"""Where the trained models are saved.

A saved model is a ``.npz`` file in :data:`ARTIFACTS_DIR` (``data/models/``, outside git); :mod:`chess_lab.models.definitions`
reads and writes it and turns it into a ``_core.Evaluator``.
"""

from pathlib import Path
from typing import Optional, Tuple

from chess_engine.paths import ROOT

#: Folder of the trained model files (ignored by git: the model cards are the record).
ARTIFACTS_DIR = ROOT / "data" / "models"


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
