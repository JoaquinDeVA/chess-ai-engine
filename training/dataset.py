"""Loading, splitting and encoding of the Stockfish evaluation dataset."""

import zlib
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from models.features import FeatureExtractor

from .targets import MAX_MATE_DISTANCE, row_targets

#: Columns every evaluations CSV must have.
REQUIRED_COLUMNS = ("fen", "cp", "is_mate", "mate_distance")

TRAIN, VALIDATION, TEST = 0, 1, 2

#: Default benchmark split: 10% test, 10% validation, 80% train.
DEFAULT_TEST_FRACTION = 0.10
DEFAULT_VALIDATION_FRACTION = 0.10
#: Salt of the split hash. Changing it changes the benchmark: keep it fixed.
SPLIT_SEED = 42


def load_evaluations(
    path: Union[str, Path],
    max_rows: Optional[int] = None,
    max_mate_distance: int = MAX_MATE_DISTANCE,
) -> pd.DataFrame:
    """Read an evaluations CSV and attach the normalized target.

    :param path: CSV with columns ``fen, cp, is_mate, mate_distance`` (White's point of view).
    :param max_rows: Keep only the first ``max_rows`` rows (before filtering), for quick runs.
    :param max_mate_distance: Mates farther than this are dropped.
    :return: DataFrame with columns ``fen`` and ``y``; rows without a valid target are removed.
    :raises ValueError: If a required column is missing.
    """
    frame = pd.read_csv(path, nrows=max_rows)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; expected {list(REQUIRED_COLUMNS)}")

    frame["is_mate"] = frame["is_mate"].astype(str).str.lower().isin(("true", "1", "1.0"))
    frame["y"] = row_targets(
        frame["is_mate"].to_numpy(),
        frame["cp"].to_numpy(),
        frame["mate_distance"].to_numpy(),
        max_mate_distance,
    )
    frame = frame.dropna(subset=["y"]).drop_duplicates(subset="fen")
    return frame[["fen", "y"]].reset_index(drop=True)


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


def split_dataset(
    frame: pd.DataFrame,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = SPLIT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataset into train, validation and test frames (see :func:`assign_split`).

    :param frame: DataFrame with a ``fen`` column.
    :param test_fraction: Share of positions in the test split.
    :param validation_fraction: Share of positions in the validation split.
    :param seed: Salt of the hash.
    :return: ``(train, validation, test)``.
    """
    split = assign_split(frame["fen"], test_fraction, validation_fraction, seed)
    return frame[split == TRAIN], frame[split == VALIDATION], frame[split == TEST]


def encode_positions(
    extractor: FeatureExtractor, fens: pd.Series, n_jobs: int = 1, chunk_size: int = 20_000
) -> Tuple[np.ndarray, ...]:
    """Encode FEN strings with ``extractor``, optionally in parallel.

    :param extractor: Board encoding to apply.
    :param fens: FEN strings.
    :param n_jobs: Worker processes (``-1`` = all cores, ``1`` = no parallelism).
    :param chunk_size: Positions per work item when running in parallel.
    :return: One stacked array per model input.
    """
    fens = list(fens)
    if n_jobs == 1 or len(fens) <= chunk_size:
        return extractor.encode_fens(fens)

    from joblib import Parallel, delayed

    chunks = [fens[start : start + chunk_size] for start in range(0, len(fens), chunk_size)]
    parts = Parallel(n_jobs=n_jobs)(delayed(extractor.encode_fens)(chunk) for chunk in chunks)
    return tuple(np.concatenate(column) for column in zip(*parts, strict=True))
