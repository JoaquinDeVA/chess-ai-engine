"""A sorted index of position keys by hash, with one value per key.

The hash is Python's string hash: it is only valid inside the process that built the index, so an
index is never saved to disk.
"""

from collections.abc import Sequence

import numpy as np


def hash_keys(keys: Sequence[str]) -> np.ndarray:
    """The ``int64`` hash of each key.

    :param keys: Position keys.
    :return: One hash per key.
    """
    return np.fromiter((hash(key) for key in keys), np.int64, count=len(keys))


def sorted_index(
    hashes: Sequence[np.ndarray], values: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Join the per-shard hashes and values and sort them by hash.

    :param hashes: Hashes of each shard.
    :param values: Values of each shard, in the same order.
    :return: ``(hashes, values)`` sorted by hash.
    """
    all_hashes, all_values = np.concatenate(hashes), np.concatenate(values)
    order = np.argsort(all_hashes, kind="stable")
    return all_hashes[order], all_values[order]


def lookup(hashes: np.ndarray, values: np.ndarray, keys: Sequence[str], missing) -> np.ndarray:
    """The value of each key that is in the index, or ``missing``.

    :param hashes: Sorted hashes of the index.
    :param values: Values in the same order.
    :param keys: Position keys to look up.
    :param missing: Value for a key that is not indexed.
    :return: One value per key.
    """
    wanted = hash_keys(keys)
    if len(hashes) == 0:
        return np.full(len(keys), missing)
    positions = np.minimum(np.searchsorted(hashes, wanted), len(hashes) - 1)
    return np.where(hashes[positions] == wanted, values[positions], missing)
