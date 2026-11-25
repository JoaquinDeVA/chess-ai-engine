"""Atomic writes: a file is complete under its final name or it is not there at all."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """A temporary name to write to; on leaving without an error it is renamed to ``path``.

    A run cut while writing leaves the previous ``path`` (or none) and a stray ``.tmp`` file, never a
    truncated file under the real name. The parent folder is created if missing.

    :param path: Final file.
    :return: The temporary path (``path`` plus ``.tmp``) the caller writes to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        yield temporary
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
