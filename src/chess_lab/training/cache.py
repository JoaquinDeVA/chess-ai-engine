"""Cache of the encoded training batches: where it lives and which directory a run uses.

Encoding half a million rows takes minutes and reading them back from the cache takes seconds, so
the batches of a run are kept in data/cache/<hash>. Every network with the same feature set shares
it, whatever its width or perspective: the encoding depends only on the inputs.
"""

import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from chess_lab.atomic import atomic_path
from chess_lab.datasets.sidecar import data_fingerprint
from chess_lab.datasets.split import DEFAULT_TEST_FRACTION, DEFAULT_VALIDATION_FRACTION
from chess_lab.defaults import CACHE_ROOT, SPLIT_SEED
from chess_lab.models.target import Representation

#: Version of the encoded-batch cache; bump it when the encoding changes without a visible key.
#: 2: the key holds the fingerprint of the data and the split.
CACHE_VERSION = 2

#: One encoded frame: ``rows, cols, values, targets`` (see ``epochs.encode_frame``).
EncodedFrame = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def cache_path(
    model_class: type,
    rep: Representation,
    positions: int,
    seed: int,
    quiet_only: bool,
    batch_size: int,
    raw_dir: Path,
    sidecar_dir: Path,
    root: Path = CACHE_ROOT,
) -> Path:
    """Directory of the encoded batches: a short hash of everything that shapes them.

    Besides the encoding, the key holds the fingerprint of the shards and sidecars and the split
    (seed and fractions): other data or another split select other rows.

    :return: ``root/<hash>``.
    """
    data = data_fingerprint(raw_dir, sidecar_dir)
    key = "|".join(
        str(part)
        for part in (
            model_class.feature_set,
            rep.name,
            positions,
            seed,
            quiet_only,
            rep.mate_keep,
            batch_size,
            data["shards"],
            data["sha12"],
            SPLIT_SEED,
            DEFAULT_TEST_FRACTION,
            DEFAULT_VALIDATION_FRACTION,
            CACHE_VERSION,
        )
    )
    return root / hashlib.sha256(key.encode()).hexdigest()[:16]


def load_frame(cache_dir: Optional[Path], index: int) -> Optional[EncodedFrame]:
    """Encoded frame ``index`` read from the cache, or None if it is not there."""
    if cache_dir is None:
        return None
    path = cache_dir / f"{index:07d}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return data["rows"], data["cols"], data["values"], data["targets"]


def save_frame(cache_dir: Path, index: int, encoded: EncodedFrame) -> None:
    """Write an encoded frame to the cache, atomically (temporary name, then rename)."""
    path = cache_dir / f"{index:07d}.npz"
    rows, cols, values, targets = encoded
    with atomic_path(path) as temporary, open(temporary, "wb") as handle:
        np.savez(handle, rows=rows, cols=cols, values=values, targets=targets)
