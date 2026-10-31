"""Splitting and encoding of the Stockfish evaluation dataset."""

import zlib

import numpy as np
import pandas as pd

TRAIN, VALIDATION, TEST = 0, 1, 2

#: Default benchmark split: 10% test, 10% validation, 80% train.
DEFAULT_TEST_FRACTION = 0.10
DEFAULT_VALIDATION_FRACTION = 0.10
#: Salt of the split hash. Changing it changes the benchmark: keep it fixed.
SPLIT_SEED = 42


def assign_split(
    fens: pd.Series,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = SPLIT_SEED,
) -> np.ndarray:
    """Assign every position to train / validation / test by hashing its FEN.

    The assignment depends only on the position itself, so it does not change with the row
    order, with ``max_rows`` or with the size of the file: a model can never be evaluated on
    a position it was trained on, whichever subset of the data was used for training.

    :param fens: FEN strings.
    :param test_fraction: Share of positions in the test split.
    :param validation_fraction: Share of positions in the validation split.
    :param seed: Salt of the hash.
    :return: Integer array with :data:`TRAIN`, :data:`VALIDATION` or :data:`TEST` per position.
    """
    buckets = np.fromiter(
        (zlib.crc32(f"{seed}:{fen}".encode()) % 10_000 for fen in fens),
        dtype=np.int64,
        count=len(fens),
    )
    test_limit = int(test_fraction * 10_000)
    validation_limit = test_limit + int(validation_fraction * 10_000)
    split = np.full(len(fens), TRAIN, dtype=np.int8)
    split[buckets < validation_limit] = VALIDATION
    split[buckets < test_limit] = TEST
    return split
