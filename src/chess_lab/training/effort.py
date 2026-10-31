"""Effort-based training of linear models: one number (0-100) sets data size, epochs and search.

:func:`effort_config` maps the effort to a configuration; :func:`calibrate` times a short fit to
estimate the duration; :func:`search` runs the random hyper-parameter trials; :func:`fit` streams
the TRAIN split of the raw shards through :meth:`LinearEvalModel.fit_stream`.
"""

import hashlib
import math
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import chess
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from chess_engine.models.linear_eval import (
    DEFAULT_FIT_CONFIG,
    LinearEvalModel,
    _losses,
    _scores,
    encode_frame,
)
from chess_engine.models.neural_eval import NeuralEvalModel
from chess_engine.models.representation import (
    Representation,
    keep_mate_rows,
    validate_mate_keep,
)
from chess_engine.models.spec import SAMPLE_SEED

from .dataset import TRAIN
from .sidecar import iter_batches

MAX_EFFORT = 100
#: Version of the encoded-batch cache; bump it when the encoding changes without a visible key.
CACHE_VERSION = 1
CACHE_ROOT = Path("data/cache/linear")
#: Positions per effort point: effort 30 trains on 30 M positions, effort 70 on 70 M.
POSITIONS_PER_EFFORT = 1_000_000
#: Floor of the positions, so that effort 0 still trains on a meaningful sample.
MIN_POSITIONS = 1_000_000
BATCH_SIZE = 4096
#: Positions of the timed calibration run (one epoch).
TIMED_POSITIONS = 50_000
#: Share of ``positions`` used by each hyper-parameter trial.
TRIAL_FRACTION = 0.1
L2_RANGE = (1e-7, 1e-3)
LR_RANGE = (1e-3, 3e-2)
#: Featurization processes of the real runs (leaves two cores free).
WORKERS = max(1, (os.cpu_count() or 1) - 2)


def effort_config(effort: int, available: int) -> Dict[str, int]:
    """Training configuration for an effort level.

    Positions are ``effort * POSITIONS_PER_EFFORT`` (effort 30 is 30 M), with a floor of
    ``MIN_POSITIONS`` and never more than ``available``. Epochs grow with effort and patience is
    fixed; hyper-parameter trials start at effort 40.

    :param effort: Effort level, ``0..100``.
    :param available: Number of TRAIN rows available.
    :return: Keys ``positions``, ``max_epochs``, ``patience``, ``trials``, ``batch_size``.
    :raises ValueError: If ``effort`` is outside ``0..100``.
    """
    if not 0 <= effort <= MAX_EFFORT:
        raise ValueError(f"effort must be between 0 and {MAX_EFFORT}, got {effort}")
    target = effort * POSITIONS_PER_EFFORT
    trials = 0 if effort < 40 else 1 + (effort - 40) // 15
    return {
        "positions": min(available, max(MIN_POSITIONS, target)),
        "max_epochs": 1 + effort // 10,
        "patience": 2,
        "trials": trials,
        "batch_size": BATCH_SIZE,
    }


SIDECAR_SUFFIX = ".sidecar.parquet"


def available_rows(
    sidecar_dir: Path,
    quiet_only: bool = False,
    raw_dir: Path | None = None,
    mate_keep: float = 1.0,
) -> int:
    """Number of TRAIN rows the stream yields: the sidecars, then the mate undersampling.

    Only the ``split`` and ``quiet`` columns are read from the sidecars. With ``mate_keep < 1``
    the ``fen`` and ``mate`` columns of the raw shards are read too, and the mates are counted
    after the same :func:`~.representation.keep_mate_rows` filter the stream applies.

    :param sidecar_dir: Directory with the ``part-*.sidecar.parquet`` files.
    :param quiet_only: Count only the quiet positions.
    :param raw_dir: Directory with the raw shards; required when ``mate_keep < 1``.
    :param mate_keep: Share of the mate positions kept, in ``(0, 1]``.
    :return: The row count (0 if there are no sidecars).
    :raises ValueError: If ``mate_keep`` is outside ``(0, 1]``, or it is below 1 without
        ``raw_dir``.
    """
    validate_mate_keep(mate_keep)
    if mate_keep < 1.0 and raw_dir is None:
        raise ValueError("raw_dir is needed to count the mates when mate_keep < 1")
    total = 0
    columns = ["split", "quiet"] if quiet_only else ["split"]
    for path in sorted(sidecar_dir.glob(f"part-*{SIDECAR_SUFFIX}")):
        table = pq.read_table(path, columns=columns)
        mask = table["split"].to_numpy() == TRAIN
        if quiet_only:
            mask &= table["quiet"].to_numpy(zero_copy_only=False)
        if mate_keep < 1.0:
            raw_path = raw_dir / (path.name.removesuffix(SIDECAR_SUFFIX) + ".parquet")
            raw = pq.read_table(raw_path, columns=["fen", "mate"]).to_pandas()
            is_mate = raw["mate"].notna().to_numpy()
            keep = np.ones(len(mask), dtype=bool)
            mates = np.flatnonzero(is_mate)
            keep[mates] = keep_mate_rows(raw["fen"].to_numpy(dtype=object)[mates], mate_keep)
            mask &= keep
        total += int(mask.sum())
    return total


class Timing(NamedTuple):
    """Measured costs of a model on this machine, used by :func:`estimated_seconds`.

    :param seconds_per_position: Training time per position of one epoch.
    :param encode_seconds: Featurizing the validation set (paid once per fit).
    :param score_seconds: Scoring the validation set (paid once per epoch).
    """

    seconds_per_position: float
    encode_seconds: float
    score_seconds: float


class _Epochs:
    """Re-iterable stream: every iteration starts a new pass over the same sample."""

    def __init__(self, make: Callable[[], Iterator[pd.DataFrame]]):
        self._make = make

    def __iter__(self) -> Iterator[pd.DataFrame]:
        return self._make()


def fit_config(
    plan: Dict[str, int],
    l2: float,
    lr: float,
    lr_decay: Optional[float] = None,
    min_lr: float = 1e-5,
) -> Dict[str, Any]:
    """Keyword configuration of :meth:`LinearEvalModel.fit_stream` for a plan.

    :param plan: Output of :func:`effort_config`.
    :param l2: L2 penalty.
    :param lr: Adam learning rate.
    :param lr_decay: Factor applied to the rate on each epoch without improvement (None: stop by
        patience instead).
    :param min_lr: With ``lr_decay``, the run stops when the rate falls below this value.
    :return: Keys ``l2, lr, batch_size, max_epochs, patience, workers, lr_decay, min_lr``.
    """
    return {
        "l2": l2,
        "lr": lr,
        "batch_size": plan["batch_size"],
        "max_epochs": plan["max_epochs"],
        "patience": plan["patience"],
        "workers": WORKERS,
        "lr_decay": lr_decay,
        "min_lr": min_lr,
    }


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


def fit(
    model_class: type,
    rep: Representation,
    raw_dir: Path,
    sidecar_dir: Path,
    validation: pd.DataFrame,
    positions: int,
    config: Dict[str, Any],
    quiet_only: bool = False,
    seed: int = SAMPLE_SEED,
    use_cache: bool = True,
) -> Tuple[LinearEvalModel, Dict[str, Any]]:
    """Fit a new model on ``positions`` TRAIN rows of the raw shards.

    :param model_class: A concrete :class:`LinearEvalModel` subclass.
    :param rep: Representation of the target.
    :param raw_dir: Directory with the raw shards.
    :param sidecar_dir: Directory with their sidecars.
    :param validation: Validation rows, for early stopping.
    :param positions: TRAIN rows used for each epoch.
    :param config: Fit configuration (see :func:`fit_config`).
    :param quiet_only: Train on quiet positions only.
    :param seed: Seed of the TRAIN sample (the same seed gives the same rows).
    :param use_cache: Keep the encoded batches in ``data/cache/linear`` and reuse them.
    :return: The fitted model and the summary of :meth:`LinearEvalModel.fit_stream`. The TRAIN
        rows are undersampled in mates by ``rep.mate_keep``.
    """
    model = model_class(rep=rep)
    batches = _Epochs(
        lambda: iter_batches(
            raw_dir,
            sidecar_dir,
            TRAIN,
            config["batch_size"],
            positions,
            seed,
            quiet_only=quiet_only,
            mate_keep=rep.mate_keep,
        )
    )
    cache = None
    if use_cache:
        cache = cache_path(model_class, rep, positions, seed, quiet_only, config["batch_size"])
    summary = model.fit_stream(batches, validation, config, cache_dir=cache)
    return model, summary


def calibrate(
    model_class: type,
    rep: Representation,
    raw_dir: Path,
    sidecar_dir: Path,
    validation: pd.DataFrame,
    plan: Dict[str, int],
    quiet_only: bool = False,
) -> Timing:
    """Time a one-epoch fit and split its cost into training and validation.

    The fit uses ``min(TIMED_POSITIONS, positions)`` training rows and the whole validation set,
    as the real runs do. Featurizing and scoring the validation set are timed separately, and the
    rest of the fit time is attributed to training.

    :param model_class: A concrete :class:`LinearEvalModel` subclass.
    :param rep: Representation of the target.
    :param raw_dir: Directory with the raw shards.
    :param sidecar_dir: Directory with their sidecars.
    :param validation: Validation rows (as used by the real runs).
    :param plan: Output of :func:`effort_config`.
    :param quiet_only: Train on quiet positions only.
    :return: The measured :class:`Timing`.
    """
    config = fit_config(plan, DEFAULT_FIT_CONFIG["l2"], DEFAULT_FIT_CONFIG["lr"])
    config["max_epochs"] = 1
    rows = min(TIMED_POSITIONS, plan["positions"])
    model, summary = fit(
        model_class, rep, raw_dir, sidecar_dir, validation, rows, config, quiet_only
    )

    started = time.perf_counter()
    rows, cols, vals, targets = encode_frame(model_class.features, rep, validation)
    encode_seconds = time.perf_counter() - started
    if isinstance(model, NeuralEvalModel):
        boards = [chess.Board(fen) for fen in validation["fen"]]
        started = time.perf_counter()
        model.predict_cp_batch(boards)
    else:
        theta = np.append(model.weights, model.bias)
        started = time.perf_counter()
        _losses(_scores(rows, cols, vals, len(targets), theta), targets).mean()
    score_seconds = time.perf_counter() - started

    train_seconds = max(0.0, summary["seconds"] - encode_seconds - score_seconds)
    return Timing(
        seconds_per_position=train_seconds / max(1, summary["train_rows"]),
        encode_seconds=encode_seconds,
        score_seconds=score_seconds,
    )


#: Seconds per position and epoch to fit on cached features (measured: 2.6 s per 500,000 rows).
CACHED_SECONDS_PER_ROW = 2.6 / 500_000


def estimated_seconds(plan: Dict[str, int], timing: Timing, cached: bool = False) -> float:
    """Estimated duration of the trials and the final fit.

    The trials refit on ``TRIAL_FRACTION`` of the positions, but each of them featurizes and scores
    the full validation set. Early stopping can make the real run shorter.

    :param plan: Output of :func:`effort_config`.
    :param timing: Output of :func:`calibrate`.
    :param cached: True when the batches are cached (only the first epoch encodes).
    :return: Training part ``(1 + trials * TRIAL_FRACTION) * positions * epochs * s_per_position``
        plus validation part ``(1 + trials) * (encode + epochs * score)``.
    """
    fits = 1 + plan["trials"]
    if cached:
        # The first epoch encodes and stores the batches; later epochs only fit on the cache.
        per_position = (
            timing.seconds_per_position + (plan["max_epochs"] - 1) * CACHED_SECONDS_PER_ROW
        )
        training = (1 + plan["trials"] * TRIAL_FRACTION) * plan["positions"] * per_position
    else:
        training = (
            (1 + plan["trials"] * TRIAL_FRACTION)
            * plan["positions"]
            * plan["max_epochs"]
            * timing.seconds_per_position
        )
    validation = fits * (timing.encode_seconds + plan["max_epochs"] * timing.score_seconds)
    return training + validation


def search(
    model_class: type,
    rep: Representation,
    raw_dir: Path,
    sidecar_dir: Path,
    validation: pd.DataFrame,
    plan: Dict[str, int],
    quiet_only: bool = False,
    log: Callable[[str], None] = print,
) -> Dict[str, float]:
    """Random search over ``l2`` and ``lr``, scored by the best validation loss.

    Each trial fits on ``TRIAL_FRACTION`` of ``plan["positions"]``.

    :param model_class: A concrete :class:`LinearEvalModel` subclass.
    :param rep: Representation of the target.
    :param raw_dir: Directory with the raw shards.
    :param sidecar_dir: Directory with their sidecars.
    :param validation: Validation rows.
    :param plan: Output of :func:`effort_config`.
    :param quiet_only: Train on quiet positions only.
    :param log: Receives one line per trial.
    :return: The best ``{"l2": ..., "lr": ...}``; the defaults if there are no trials.
    """
    best = {"l2": DEFAULT_FIT_CONFIG["l2"], "lr": DEFAULT_FIT_CONFIG["lr"]}
    if plan["trials"] == 0:
        return best
    rng = np.random.default_rng(SAMPLE_SEED)
    rows = max(1, int(plan["positions"] * TRIAL_FRACTION))
    best_loss = math.inf
    for trial in range(1, plan["trials"] + 1):
        candidate = {
            "l2": _log_uniform(rng, *L2_RANGE),
            "lr": _log_uniform(rng, *LR_RANGE),
        }
        _, summary = fit(
            model_class,
            rep,
            raw_dir,
            sidecar_dir,
            validation,
            rows,
            fit_config(plan, candidate["l2"], candidate["lr"]),
            quiet_only,
        )
        loss = min(summary["val_losses"])
        log(
            f"trial {trial}/{plan['trials']}: l2={candidate['l2']:.2e} "
            f"lr={candidate['lr']:.2e} validation loss {loss:.5f}"
        )
        if loss < best_loss:
            best, best_loss = candidate, loss
    return best


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """Sample uniformly on a log scale in ``[low, high]``."""
    return math.exp(rng.uniform(math.log(low), math.log(high)))
