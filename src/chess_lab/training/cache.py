"""Cache of the encoded training batches: where it lives and which directory a run uses.

Encoding half a million rows takes minutes and reading them back from the cache takes seconds, so
the batches of a run are kept in data/cache/<hash>. Every network shares it.
"""

import hashlib
from pathlib import Path

from chess_lab.models.target import Representation

#: Version of the encoded-batch cache; bump it when the encoding changes without a visible key.
CACHE_VERSION = 1
CACHE_ROOT = Path("data/cache")


def cache_path(
    model_class: type,
    rep: Representation,
    positions: int,
    seed: int,
    quiet_only: bool,
    batch_size: int,
    root: Path = CACHE_ROOT,
) -> Path:
    """Directory of the encoded batches: a short hash of everything that shapes them.

    :return: ``root/<hash>``.
    """
    key = "|".join(
        str(part)
        for part in (
            model_class.__name__,
            rep.name,
            positions,
            seed,
            quiet_only,
            rep.mate_keep,
            batch_size,
            CACHE_VERSION,
        )
    )
    return root / hashlib.sha256(key.encode()).hexdigest()[:16]
