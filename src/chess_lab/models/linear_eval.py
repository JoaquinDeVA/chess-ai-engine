"""Training of the linear models: encoding, Adam fit and the epoch loop shared with the network.

The saved model holds the weights and the feature set name; the Python feature extractors, the targets and the fit live here.
"""

import functools
import math
import multiprocessing
import os
import time
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import chess
import numpy as np
import pandas as pd

from .definitions import LinearEvalModel
from .linear_features import LinearFeatures, features_of
from .representation import Representation, target

#: Hyper-parameters used when the training configuration leaves a key out.
DEFAULT_FIT_CONFIG: Dict[str, Any] = {
    "l2": 1e-6,
    "lr": 0.01,
    "batch_size": 4096,
    "max_epochs": 20,
    "patience": 3,
    "workers": 1,
}
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPS = 1e-8
#: Thread-count variables pinned to 1 in the featurization workers (the BLAS threads of
#: numpy would otherwise oversubscribe the cores).
_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

EncodedFrame = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def new_linear_model(model_class: type, rep: Optional[Representation] = None) -> LinearEvalModel:
    """An untrained model: zero weights, sized for the feature set under the perspective.

    :param model_class: A concrete :class:`LinearEvalModel` subclass.
    :param rep: Representation of the target; defaults to :class:`Representation`.
    :return: The model, with ``rep`` saved as a dictionary.
    """
    rep = rep if rep is not None else Representation()
    size = features_of(model_class).size(rep.perspective)
    return model_class(weights=np.zeros(size), rep=rep.to_dict())


def fit_stream(
    model: LinearEvalModel,
    batches: Iterable[pd.DataFrame],
    validation: pd.DataFrame,
    config: Dict[str, Any],
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train with Adam on the binary cross-entropy against soft targets.

    ``batches`` is read once per epoch, so it must be re-iterable (a list, or an object whose
    ``__iter__`` restarts), not a one-shot generator. Minibatches never span two frames: each
    frame is cut into chunks of at most ``batch_size`` rows, so the effective batch size
    depends on how the caller chunks the data. The weights of the epoch with the lowest
    validation loss are kept.

    With ``workers > 1`` the frames are featurized in a process pool, in order, while the
    main process runs the updates; the result is the same as with one worker.

    :param model: The model to train; its weights and bias are replaced by the best epoch.
    :param batches: DataFrames with columns ``fen, cp, mate, white_to_move``.
    :param validation: Same columns, used for early stopping; must not be empty.
    :param config: Keys ``l2``, ``lr``, ``batch_size``, ``max_epochs``, ``patience``,
        ``workers``; missing keys take :data:`DEFAULT_FIT_CONFIG`. Optional ``lr_decay`` in
        (0, 1) and ``min_lr`` (default 1e-5): an epoch without validation improvement
        multiplies the learning rate by ``lr_decay`` and the run goes on (``patience`` is not
        used) until the rate falls below ``min_lr`` or ``max_epochs`` is reached.
    :param cache_dir: If set, the encoded frames are stored there (``<index>.npz``) and read
        back instead of being encoded again; the caller must key the directory by everything
        that determines the frames. None encodes every epoch.
    :return: Summary with ``epochs``, ``best_epoch``, ``train_losses``, ``val_losses``,
        ``lr_history`` (rate of each epoch),
        ``train_rows``, ``seconds`` and the ``config`` used.
    :raises ValueError: If an epoch yields no rows, the validation set is empty, or a
        validation loss is not finite.
    """
    cfg = {**DEFAULT_FIT_CONFIG, **config}
    started = time.perf_counter()
    if len(validation) == 0:
        raise ValueError("validation set is empty; early stopping needs validation rows")
    features, rep = features_of(model), Representation.from_dict(model.rep)
    size = model.weights.size
    theta = np.append(model.weights, model.bias)  # coefficients, then the intercept
    moment, velocity = np.zeros_like(theta), np.zeros_like(theta)
    step = 0
    validation_data = encode_frame(features, rep, validation)
    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])

    def train_batch(encoded: EncodedFrame) -> Tuple[float, int]:
        nonlocal moment, velocity, step
        loss_sum, rows_seen = 0.0, 0
        for local in _minibatches(*encoded, batch_size):
            local_rows, local_cols, local_vals, local_y = local
            count = len(local_y)
            z = _scores(local_rows, local_cols, local_vals, count, theta)
            loss_sum += float(np.sum(_losses(z, local_y)))
            rows_seen += count
            grad_z = 0.5 * (1.0 + np.tanh(0.5 * z)) - local_y
            grad_w = (
                np.bincount(
                    local_cols,
                    weights=local_vals * grad_z[local_rows],
                    minlength=size,
                )
                / count
                + float(cfg["l2"]) * theta[:size]
            )
            grad = np.append(grad_w, grad_z.mean())
            step += 1
            moment = _ADAM_BETA1 * moment + (1 - _ADAM_BETA1) * grad
            velocity = _ADAM_BETA2 * velocity + (1 - _ADAM_BETA2) * grad * grad
            m_hat = moment / (1 - _ADAM_BETA1**step)
            v_hat = velocity / (1 - _ADAM_BETA2**step)
            theta[:] -= lr * m_hat / (np.sqrt(v_hat) + _ADAM_EPS)
        return loss_sum, rows_seen

    def validation_loss() -> float:
        val_rows, val_cols, val_vals, val_y = validation_data
        val_z = _scores(val_rows, val_cols, val_vals, len(val_y), theta)
        return float(np.mean(_losses(val_z, val_y)))

    def set_lr(value: float) -> None:
        nonlocal lr
        lr = value

    summary, best_theta = run_epochs(
        cfg=cfg,
        started=started,
        frames=lambda executor: encoded_frames(
            features, rep, batches, int(cfg["workers"]), executor, cache_dir
        ),
        train_batch=train_batch,
        validation_loss=validation_loss,
        get_lr=lambda: lr,
        set_lr=set_lr,
        snapshot=theta.copy,
        initial_best=theta.copy(),
    )
    model.weights = best_theta[:size]
    model.bias = float(best_theta[size])
    return summary


def encoded_frames(
    features: LinearFeatures,
    rep: Representation,
    batches: Iterable[pd.DataFrame],
    workers: int,
    executor: Optional[Executor],
    cache_dir: Optional[Path] = None,
) -> Iterator[EncodedFrame]:
    """Encoded frames in order, featurized by the pool when there is one.

    At most ``2 * workers`` frames are in flight, so memory stays bounded. With a cache, a
    frame already stored is read instead of encoded, and a new one is stored once encoded.

    :param features: Feature set.
    :param rep: Representation of the target.
    :param batches: The frames of one epoch.
    :param workers: Number of workers (``1`` means no pool).
    :param executor: The pool, or None.
    :param cache_dir: Directory of the encoded frames, or None for no cache.
    :return: Iterator of :data:`EncodedFrame`, in the order of ``batches``.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    encode = functools.partial(encode_frame, features, rep)
    window = 2 * workers
    pending: deque = deque()  # (index, future)
    for index, frame in enumerate(batches):
        cached = _load_encoded(cache_dir, index)
        if cached is not None:
            pending.append((index, cached))
        elif executor is None:
            encoded = encode(frame)
            if cache_dir is not None:
                _save_encoded(cache_dir, index, encoded)
            pending.append((index, encoded))
        else:
            pending.append((index, executor.submit(encode, frame)))
        if len(pending) > window:
            yield _resolve(pending.popleft(), cache_dir)
    while pending:
        yield _resolve(pending.popleft(), cache_dir)


def run_epochs(
    *,
    cfg: Dict[str, Any],
    started: float,
    frames: Callable[[Optional[Executor]], Iterable[EncodedFrame]],
    train_batch: Callable[[EncodedFrame], Tuple[float, int]],
    validation_loss: Callable[[], float],
    get_lr: Callable[[], float],
    set_lr: Callable[[float], None],
    snapshot: Callable[[], Any],
    initial_best: Any,
) -> Tuple[Dict[str, Any], Any]:
    """Epoch loop shared by the models: encoded frames, decay or patience, best-epoch state.

    Each model supplies how an encoded frame is trained and how the validation loss is computed.
    With ``lr_decay`` in ``cfg`` an epoch without improvement multiplies the learning rate and the
    run stops when it falls below ``min_lr``; otherwise it stops after more than ``patience``
    epochs without improvement.

    :param cfg: Fit configuration (``max_epochs``, ``workers``, ``patience``, ``lr_decay``...).
    :param started: ``time.perf_counter()`` at the start of the fit, for ``seconds``.
    :param frames: Given the pool (or None), the encoded frames of one epoch.
    :param train_batch: Trains on one encoded frame; returns ``(loss_sum, rows)``.
    :param validation_loss: Mean validation loss with the current weights.
    :param get_lr: Current learning rate.
    :param set_lr: Sets the learning rate.
    :param snapshot: State to keep when an epoch is the best so far.
    :param initial_best: State to return if no epoch improves on infinity.
    :return: ``(summary, best_state)``; the summary has ``epochs``, ``best_epoch``,
        ``train_losses``, ``val_losses``, ``lr_history``, ``train_rows``, ``seconds``, ``config``.
    :raises ValueError: If an epoch yields no rows or a validation loss is not finite.
    """
    best_state, best_loss, best_epoch = initial_best, math.inf, 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    lr_history: List[float] = []
    bad_epochs = 0
    train_rows = 0
    lr_decay = cfg.get("lr_decay")
    min_lr = float(cfg.get("min_lr", 1e-5))

    with _featurization_pool(int(cfg["workers"])) as executor:
        for epoch in range(1, int(cfg["max_epochs"]) + 1):
            loss_sum, rows_seen = 0.0, 0
            lr_history.append(get_lr())
            for encoded in frames(executor):
                frame_loss, frame_rows = train_batch(encoded)
                loss_sum += frame_loss
                rows_seen += frame_rows
            if rows_seen == 0:
                raise ValueError("batches yielded no rows; pass a re-iterable collection")
            train_rows = rows_seen
            train_losses.append(loss_sum / rows_seen)

            val_loss = validation_loss()
            if not math.isfinite(val_loss):
                raise ValueError(f"validation loss is not finite at epoch {epoch}: {val_loss}")
            val_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss, best_epoch, best_state = val_loss, epoch, snapshot()
                bad_epochs = 0
            elif lr_decay is not None:
                lr = get_lr() * float(lr_decay)
                if lr < min_lr:
                    break
                set_lr(lr)
            else:
                bad_epochs += 1
                if bad_epochs > int(cfg["patience"]):
                    break

    summary = {
        "epochs": len(train_losses),
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "lr_history": lr_history,
        "train_rows": train_rows,
        "seconds": round(time.perf_counter() - started, 3),
        "config": cfg,
    }
    return summary, best_state


def _resolve(item: Tuple[int, Any], cache_dir: Optional[Path]) -> EncodedFrame:
    """The encoded frame of a pending item; a future is awaited and its result cached."""
    index, value = item
    if isinstance(value, tuple):
        return value
    encoded = value.result()
    if cache_dir is not None:
        _save_encoded(cache_dir, index, encoded)
    return encoded


def _load_encoded(cache_dir: Optional[Path], index: int) -> Optional[EncodedFrame]:
    """Encoded frame ``index`` read from the cache, or None if it is not there."""
    if cache_dir is None:
        return None
    path = cache_dir / f"{index:07d}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return data["rows"], data["cols"], data["values"], data["targets"]


def _save_encoded(cache_dir: Path, index: int, encoded: EncodedFrame) -> None:
    """Write an encoded frame to the cache, atomically (temporary name, then rename)."""
    path = cache_dir / f"{index:07d}.npz"
    tmp = cache_dir / f"{index:07d}.tmp"
    rows, cols, values, targets = encoded
    with open(tmp, "wb") as handle:
        np.savez(handle, rows=rows, cols=cols, values=values, targets=targets)
    os.replace(tmp, path)


def encode_frame(
    features: LinearFeatures, rep: Representation, frame: pd.DataFrame
) -> EncodedFrame:
    """Sparse features of the rows of a frame, with their soft targets.

    Module-level so that featurization workers can run it.

    :param features: Feature set.
    :param rep: Representation; its perspective selects the features, its scale the targets.
    :param frame: Rows with ``fen, cp, mate, white_to_move``.
    :return: ``(rows, cols, values, targets)``; ``rows`` is sorted and indexes ``targets``.
    """
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for row, fen in enumerate(frame["fen"]):
        indices, values = features.active(chess.Board(fen), rep.perspective)
        rows.extend([row] * len(indices))
        cols.extend(indices)
        vals.extend(values)
    y = target(
        np.asarray(frame["cp"], dtype=np.float64),
        np.asarray(frame["mate"], dtype=np.float64),
        np.asarray(frame["white_to_move"], dtype=bool),
        rep,
    )
    return (
        np.asarray(rows, dtype=np.intp),
        np.asarray(cols, dtype=np.intp),
        np.asarray(vals, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )


@contextmanager
def _featurization_pool(workers: int) -> Iterator[Optional[ProcessPoolExecutor]]:
    """Process pool for featurization, with single-threaded children.

    The thread-count variables are set while the pool starts its (spawned) workers and restored
    on exit, so the parent environment is left as it was.

    :param workers: Number of processes; ``1`` yields None (no pool).
    :return: The pool, or None.
    :raises ValueError: If ``workers`` is below 1.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    if workers == 1:
        yield None
        return
    saved = {name: os.environ.get(name) for name in _THREAD_ENV}
    os.environ.update({name: "1" for name in _THREAD_ENV})
    try:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            yield executor
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _minibatches(rows, cols, vals, y, batch_size: int):
    """Split encoded rows into consecutive minibatches.

    :param rows: Sorted row indices of the sparse entries.
    :param cols: Feature index of each entry.
    :param vals: Value of each entry.
    :param y: Targets, one per row.
    :param batch_size: Rows per minibatch.
    :return: Generator of ``(local_rows, cols, vals, targets)`` with rows renumbered from 0.
    """
    for start in range(0, len(y), batch_size):
        stop = min(len(y), start + batch_size)
        first, last = np.searchsorted(rows, [start, stop])
        yield rows[first:last] - start, cols[first:last], vals[first:last], y[start:stop]


def _scores(rows, cols, vals, count: int, theta: np.ndarray) -> np.ndarray:
    """Logits of ``count`` rows, from the sparse entries and ``theta = (weights, bias)``.

    :return: Array of shape ``(count,)``.
    """
    weights = theta[:-1]
    return np.bincount(rows, weights=vals * weights[cols], minlength=count) + theta[-1]


def _losses(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Binary cross-entropy with soft targets, from logits.

    :return: ``log(1 + exp(z)) - y * z`` per row.
    """
    return np.logaddexp(0.0, z) - y * z
