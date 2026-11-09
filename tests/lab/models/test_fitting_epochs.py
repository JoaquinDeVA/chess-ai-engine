"""Tests of the shared fit plumbing: encoding, its cache, the workers and the epoch loop."""

import os
import time

import numpy as np
import pytest

from chess_lab.models.features import FEATURE_SETS
from chess_lab.models.fitting import epochs
from chess_lab.models.target import Representation
from tests.conftest import random_positions
from tests.lab.models.positions import frame

FEATURES = FEATURE_SETS["pst_slotted"]


def _batches():
    boards = random_positions(120, seed=21)
    return [frame(boards[i : i + 40]) for i in range(0, len(boards), 40)]


def _encoded(batches, workers=1, cache_dir=None):
    with epochs._featurization_pool(workers) as executor:
        return list(
            epochs.encoded_frames(FEATURES, Representation(), batches, workers, executor, cache_dir)
        )


def _same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_encode_frame_gives_one_target_per_row_and_sorted_rows():
    rows, cols, values, targets = epochs.encode_frame(
        FEATURES, Representation(), frame(random_positions(10, seed=3))
    )
    assert len(targets) == 10
    assert np.all(np.diff(rows) >= 0) and rows.max() == 9
    assert cols.max() < FEATURES.size() and np.all(values == 1.0)


def test_the_cache_gives_the_same_frames_and_a_second_run_does_not_encode(tmp_path, monkeypatch):
    plain = _encoded(_batches())
    cached = _encoded(_batches(), cache_dir=tmp_path)
    assert all(_same(a, b) for a, b in zip(plain, cached, strict=True))
    assert len(list(tmp_path.glob("*.npz"))) == 3
    assert not list(tmp_path.glob("*.tmp"))

    def never(*_):
        raise AssertionError("encoded a batch despite the cache")

    monkeypatch.setattr(epochs, "encode_frame", never)
    again = _encoded(_batches(), cache_dir=tmp_path)
    assert all(_same(a, b) for a, b in zip(plain, again, strict=True))


def test_workers_give_the_same_frames_as_one_worker_and_restore_the_environment():
    env_before = {name: os.environ.get(name) for name in epochs._THREAD_ENV}
    serial = _encoded(_batches(), workers=1)
    parallel = _encoded(_batches(), workers=2)
    assert all(_same(a, b) for a, b in zip(serial, parallel, strict=True))
    assert {name: os.environ.get(name) for name in epochs._THREAD_ENV} == env_before


def _run(val_losses, **cfg):
    """The epoch loop on one fake frame, with the given validation loss per epoch."""
    state = {"lr": 0.1, "epoch": 0}
    losses = iter(val_losses)

    def validation_loss():
        state["epoch"] += 1
        return next(losses)

    config = {"max_epochs": len(val_losses), "workers": 1, "patience": 0, **cfg}
    return epochs.run_epochs(
        cfg=config,
        started=time.perf_counter(),
        frames=lambda executor: [None],
        train_batch=lambda encoded: (1.0, 1),
        validation_loss=validation_loss,
        get_lr=lambda: state["lr"],
        set_lr=lambda value: state.update(lr=value),
        snapshot=lambda: state["epoch"],
        initial_best=0,
    )


def test_lr_decay_keeps_training_with_a_smaller_rate_until_max_epochs():
    summary, best = _run([0.5, 0.6, 0.4, 0.45, 0.3, 0.35], lr_decay=0.5, min_lr=1e-9)
    assert summary["epochs"] == 6  # patience 0 does not cut the run
    # An epoch without improvement halves the rate for the next one; one that improves keeps it.
    assert summary["lr_history"] == pytest.approx([0.1, 0.1, 0.05, 0.05, 0.025, 0.025])
    assert best == summary["best_epoch"] == 5


def test_decay_patience_stops_after_that_many_epochs_in_a_row_without_improvement():
    summary, _ = _run(
        [0.5, 0.6, 0.4, 0.45, 0.46, 0.3], patience=2, decay_patience=True, lr_decay=0.5
    )
    assert summary["epochs"] == 5  # epochs 4 and 5 do not improve: two in a row


def test_lr_decay_stops_when_the_rate_falls_below_min_lr():
    summary, _ = _run([0.5, 0.6, 0.4], lr_decay=0.5, min_lr=0.06)
    # Epoch 2 does not improve: 0.1 * 0.5 = 0.05 < 0.06, and the run ends there.
    assert summary["epochs"] == 2
    assert summary["lr_history"] == [0.1, 0.1]


def test_without_lr_decay_patience_stops_and_the_rate_is_constant():
    summary, best = _run([0.5, 0.6, 0.7, 0.4], patience=1)
    assert summary["epochs"] == 3  # best at 1, no improvement at 2 and 3 (> patience)
    assert summary["lr_history"] == [0.1, 0.1, 0.1]
    assert best == 1


def test_a_non_finite_validation_loss_is_refused():
    with pytest.raises(ValueError, match="not finite"):
        _run([0.5, float("nan")])


def test_an_epoch_without_rows_is_refused():
    with pytest.raises(ValueError, match="no rows"):
        epochs.run_epochs(
            cfg={"max_epochs": 1, "workers": 1, "patience": 0},
            started=time.perf_counter(),
            frames=lambda executor: [],
            train_batch=lambda encoded: (0.0, 0),
            validation_loss=lambda: 0.5,
            get_lr=lambda: 0.1,
            set_lr=lambda value: None,
            snapshot=lambda: None,
            initial_best=None,
        )
