"""Effort-to-configuration curves, sidecar row counts, cost estimate and random-search trials."""

from pathlib import Path

import chess
import pandas as pd
import pytest

from chess_lab.datasets.raw_shards import shard_path, write_shard
from chess_lab.datasets.sidecar import build_sidecar, count_rows, load_sample
from chess_lab.datasets.split import TRAIN, VALIDATION, assign_split
from chess_lab.models.definitions import NnW64BothModel
from chess_lab.models.target import Representation
from chess_lab.training import effort
from chess_lab.training.effort import (
    MIN_POSITIONS,
    TRIAL_MAX_ROWS,
    Timing,
    effort_config,
    estimated_seconds,
    search,
    trial_rows,
)
from tests.conftest import material_cp, random_positions


def write_fixture_shards(raw: Path, side: Path, count: int, seed: int) -> list[str]:
    """One raw shard with ``count`` random positions, and its sidecar. Returns the FENs."""
    raw.mkdir(parents=True, exist_ok=True)
    boards = random_positions(count, seed=seed)
    fens = [" ".join(b.fen().split()[:4]) for b in boards]
    records = [
        {"fen": fen, "cp": material_cp(board), "mate": None, "depth": 20, "knodes": 1,
         "best_move": None, "n_evals": 1, "n_pvs": 1}  # fmt: skip
        for fen, board in zip(fens, boards, strict=True)
    ]  # fmt: skip
    write_shard(shard_path(raw, 0), records, {"perspective": "white"})
    build_sidecar(shard_path(raw, 0), side)
    return fens


def test_effort_bounds():
    assert effort_config(0, 10**9)["positions"] == MIN_POSITIONS
    assert effort_config(100, 10**9)["positions"] == 10**9


def test_effort_is_a_percentage_of_the_available_rows():
    assert effort_config(30, 200_000_000)["positions"] == 60_000_000
    assert effort_config(70, 200_000_000)["positions"] == 140_000_000
    assert effort_config(100, 200_000_000)["positions"] == 200_000_000


def test_effort_floors_positions_at_the_minimum():
    assert effort_config(0, 10**9)["positions"] == MIN_POSITIONS
    assert effort_config(1, 10**8)["positions"] == MIN_POSITIONS


def test_effort_is_monotonic():
    previous = None
    for effort_level in range(101):
        config = effort_config(effort_level, 10**9)
        if previous is not None:
            for key in ("positions", "trials"):
                assert config[key] >= previous[key], (effort_level, key)
        previous = config


def test_effort_caps_positions_at_available_rows():
    assert effort_config(0, 5_000)["positions"] == 5_000
    assert effort_config(100, 5_000)["positions"] == 5_000


def test_effort_fixed_settings():
    low, high = effort_config(0, 10**9), effort_config(100, 10**9)
    assert low["trials"] == 0 and high["trials"] == 5
    for plan in (low, effort_config(50, 10**9), high):  # effort does not change the recipe
        assert (plan["max_epochs"], plan["patience"]) == (100, 3)
    assert low["batch_size"] == high["batch_size"] == 4096


def test_trial_rows_are_a_tenth_capped():
    assert trial_rows(30_000_000) == 3_000_000
    assert trial_rows(190_000_000) == TRIAL_MAX_ROWS
    assert trial_rows(5) == 1


@pytest.mark.parametrize("effort_level", [-1, 101])
def test_effort_rejects_out_of_range(effort_level):
    with pytest.raises(ValueError):
        effort_config(effort_level, 10**6)


def test_count_rows_counts_exact_train_rows_from_sidecars(tmp_path):
    fens = write_fixture_shards(tmp_path / "raw", tmp_path / "side", 150, seed=5)
    train = [
        f for f, split in zip(fens, assign_split(pd.Series(fens)), strict=True) if split == TRAIN
    ]
    quiet = [f for f in train if not chess.Board(f).is_check()]
    raw, side = tmp_path / "raw", tmp_path / "side"
    assert count_rows(raw, side, TRAIN) == len(train) > 0
    assert count_rows(raw, side, TRAIN, quiet_only=True) == len(quiet)
    assert count_rows(tmp_path / "empty", side, TRAIN) == 0


def test_estimate_formula_on_fixed_timing():
    timing = Timing(seconds_per_position=1e-3, encode_seconds=2.0, score_seconds=0.5)
    # Final fit only: 1000 positions * 1 epoch * 1e-3 s, plus encode 2.0 and one score of 0.5.
    no_trials = {"positions": 1000, "max_epochs": 1, "trials": 0}
    assert estimated_seconds(no_trials, timing) == pytest.approx(1.0 + 2.0 + 0.5)
    # Two random trials and the defaults, each on 10% of 10000 positions: training
    # (1 + 0.3) * 10000 * 3 * 1e-3 = 39 s; four fits each with encode 2.0 and 3 epochs of
    # 0.5 s scoring: 4 * (2.0 + 1.5) = 14 s.
    with_trials = {"positions": 10_000, "max_epochs": 3, "trials": 2}
    assert estimated_seconds(with_trials, timing) == pytest.approx(39.0 + 14.0)


def test_search_keeps_the_trial_with_the_lowest_validation_loss(monkeypatch):
    calls = []
    losses = iter([0.9, 0.4, 0.6, 0.5])

    def fake_fit(
        model_class, rep, raw_dir, sidecar_dir, validation, positions, config, quiet=False
    ):
        calls.append(config)
        return None, {"val_losses": [next(losses)]}

    monkeypatch.setattr(effort, "fit", fake_fit)
    plan = {"positions": 1000, "max_epochs": 1, "patience": 2, "trials": 3, "batch_size": 4096}
    best = search(NnW64BothModel, Representation(), Path("."), Path("."), None, plan,
                  log=lambda line: None)  # fmt: skip
    assert len(calls) == 4  # the defaults and three random candidates
    assert (calls[0]["l2"], calls[0]["lr"]) == (1e-6, 0.01)
    assert best == {"l2": calls[1]["l2"], "lr": calls[1]["lr"]}
    assert all(1e-7 <= c["l2"] <= 1e-3 and 1e-3 <= c["lr"] <= 3e-2 for c in calls[1:])


def test_search_keeps_the_defaults_when_the_only_draw_is_worse(monkeypatch):
    losses = iter([0.3, 0.8])
    monkeypatch.setattr(effort, "fit", lambda *a, **k: (None, {"val_losses": [next(losses)]}))
    plan = {"positions": 1000, "max_epochs": 1, "patience": 2, "trials": 1, "batch_size": 4096}
    best = search(NnW64BothModel, Representation(), Path("."), Path("."), None, plan,
                  log=lambda line: None)  # fmt: skip
    assert best == {"l2": 1e-6, "lr": 0.01}


def test_trial_path_reads_only_the_train_split(tmp_path, monkeypatch):
    raw, side = tmp_path / "raw", tmp_path / "side"
    write_fixture_shards(raw, side, 900, seed=4)
    validation = load_sample(raw, side, VALIDATION, 500, seed=42)
    plan = effort_config(40, count_rows(raw, side, TRAIN))
    assert plan["trials"] == 1

    splits = []
    real_iter_batches = effort.iter_batches

    def recording_iter_batches(raw_dir, sidecar_dir, split, *args, **kwargs):
        splits.append(split)
        return real_iter_batches(raw_dir, sidecar_dir, split, *args, **kwargs)

    def reading_fit_stream(model, batches, validation, config, cache_dir=None, checkpoint=None):
        list(batches)  # reads the stream as the fit would, without training
        return {"val_losses": [0.5]}

    monkeypatch.setattr(effort, "iter_batches", recording_iter_batches)
    monkeypatch.setattr(effort, "fit_stream", reading_fit_stream)
    lines = []
    search(NnW64BothModel, Representation(), raw, side, validation, plan, log=lines.append)
    assert splits and set(splits) == {TRAIN}
    assert len(lines) == 2 and lines[1].startswith("trial 2/2")


def write_mate_fixture_shards(raw: Path, side: Path, count: int, seed: int) -> None:
    """One raw shard with ``count`` random positions, every third one a mate, and its sidecar."""
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for index, board in enumerate(random_positions(count, seed=seed)):
        record = {"fen": " ".join(board.fen().split()[:4]), "cp": material_cp(board),
                  "mate": None, "depth": 20, "knodes": 1, "best_move": None,
                  "n_evals": 1, "n_pvs": 1}  # fmt: skip
        if index % 3 == 0:
            record.update(cp=None, mate=2)
        records.append(record)
    write_shard(shard_path(raw, 0), records, {"perspective": "white"})
    build_sidecar(shard_path(raw, 0), side)


def test_count_rows_counts_mates_after_the_mate_keep_filter(tmp_path):
    raw, side = tmp_path / "raw", tmp_path / "side"
    write_mate_fixture_shards(raw, side, 300, seed=4)
    full = count_rows(raw, side, TRAIN)
    kept = count_rows(raw, side, TRAIN, mate_keep=0.2)
    assert 0 < kept < full
    # The planned count is the number of rows the TRAIN stream yields.
    assert kept == len(load_sample(raw, side, TRAIN, 10**6, mate_keep=0.2))
    assert count_rows(raw, side, TRAIN, mate_keep=1.0) == full
