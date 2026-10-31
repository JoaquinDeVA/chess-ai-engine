"""Effort-to-configuration curves, sidecar row counts, cost estimate and random-search trials."""

from pathlib import Path

import chess
import pandas as pd
import pytest

from chess_lab.models.definitions import PstTaperedModel
from chess_lab.models.representation import Representation
from chess_lab.training import effort
from chess_lab.training.dataset import TRAIN, VALIDATION, assign_split
from chess_lab.training.effort import (
    MIN_POSITIONS,
    Timing,
    available_rows,
    cache_path,
    effort_config,
    estimated_seconds,
    search,
)
from chess_lab.training.lichess_raw import shard_path, write_shard
from chess_lab.training.sidecar import build_sidecar, load_sample
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
    assert effort_config(100, 10**9)["positions"] == 100_000_000


def test_effort_is_millions_of_positions():
    assert effort_config(30, 10**9)["positions"] == 30_000_000
    assert effort_config(70, 10**9)["positions"] == 70_000_000


def test_effort_floors_positions_at_the_minimum():
    assert effort_config(0, 10**9)["positions"] == MIN_POSITIONS
    assert effort_config(5, 10**9)["positions"] == 5_000_000


def test_effort_is_monotonic():
    previous = None
    for effort_level in range(101):
        config = effort_config(effort_level, 10**9)
        if previous is not None:
            for key in ("positions", "max_epochs", "trials"):
                assert config[key] >= previous[key], (effort_level, key)
        previous = config


def test_effort_caps_positions_at_available_rows():
    assert effort_config(0, 5_000)["positions"] == 5_000
    assert effort_config(100, 5_000)["positions"] == 5_000


def test_effort_fixed_settings():
    low, high = effort_config(0, 10**9), effort_config(100, 10**9)
    assert low["trials"] == 0 and high["trials"] == 5
    assert low["max_epochs"] == 1 and high["max_epochs"] == 11
    assert low["patience"] == high["patience"] == 2
    assert low["batch_size"] == high["batch_size"] == 4096


@pytest.mark.parametrize("effort_level", [-1, 101])
def test_effort_rejects_out_of_range(effort_level):
    with pytest.raises(ValueError):
        effort_config(effort_level, 10**6)


def test_available_rows_counts_exact_train_rows_from_sidecars(tmp_path):
    fens = write_fixture_shards(tmp_path / "raw", tmp_path / "side", 150, seed=5)
    train = [
        f for f, split in zip(fens, assign_split(pd.Series(fens)), strict=True) if split == TRAIN
    ]
    quiet = [f for f in train if not chess.Board(f).is_check()]
    assert available_rows(tmp_path / "side") == len(train) > 0
    assert available_rows(tmp_path / "side", quiet_only=True) == len(quiet)
    assert available_rows(tmp_path / "empty") == 0


def test_estimate_formula_on_fixed_timing():
    timing = Timing(seconds_per_position=1e-3, encode_seconds=2.0, score_seconds=0.5)
    # Final fit only: 1000 positions * 1 epoch * 1e-3 s, plus encode 2.0 and one score of 0.5.
    no_trials = {"positions": 1000, "max_epochs": 1, "trials": 0}
    assert estimated_seconds(no_trials, timing) == pytest.approx(1.0 + 2.0 + 0.5)
    # Two trials on 10% of 10000 positions each: training (1 + 0.2) * 10000 * 3 * 1e-3 = 36 s;
    # three fits each with encode 2.0 and 3 epochs of 0.5 s scoring: 3 * (2.0 + 1.5) = 10.5 s.
    with_trials = {"positions": 10_000, "max_epochs": 3, "trials": 2}
    assert estimated_seconds(with_trials, timing) == pytest.approx(36.0 + 10.5)


def test_search_keeps_the_trial_with_the_lowest_validation_loss(monkeypatch):
    calls = []
    losses = iter([0.9, 0.4, 0.6])

    def fake_fit(
        model_class, rep, raw_dir, sidecar_dir, validation, positions, config, quiet=False
    ):
        calls.append(config)
        return None, {"val_losses": [next(losses)]}

    monkeypatch.setattr(effort, "fit", fake_fit)
    plan = {"positions": 1000, "max_epochs": 1, "patience": 2, "trials": 3, "batch_size": 4096}
    best = search(PstTaperedModel, Representation(), Path("."), Path("."), None, plan,
                  log=lambda line: None)  # fmt: skip
    assert len(calls) == 3
    assert best == {"l2": calls[1]["l2"], "lr": calls[1]["lr"]}
    assert all(1e-7 <= c["l2"] <= 1e-3 and 1e-3 <= c["lr"] <= 3e-2 for c in calls)


def test_trial_path_reads_only_the_train_split(tmp_path, monkeypatch):
    raw, side = tmp_path / "raw", tmp_path / "side"
    write_fixture_shards(raw, side, 900, seed=4)
    validation = load_sample(raw, side, VALIDATION, 500, seed=42)
    plan = effort_config(40, available_rows(side))
    assert plan["trials"] == 1

    splits = []
    real_iter_batches = effort.iter_batches

    def recording_iter_batches(raw_dir, sidecar_dir, split, *args, **kwargs):
        splits.append(split)
        return real_iter_batches(raw_dir, sidecar_dir, split, *args, **kwargs)

    monkeypatch.setattr(effort, "iter_batches", recording_iter_batches)
    lines = []
    search(PstTaperedModel, Representation(), raw, side, validation, plan, log=lines.append)
    assert splits and set(splits) == {TRAIN}
    assert len(lines) == 1 and lines[0].startswith("trial 1/1")


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


def test_available_rows_counts_mates_after_the_mate_keep_filter(tmp_path):
    raw, side = tmp_path / "raw", tmp_path / "side"
    write_mate_fixture_shards(raw, side, 900, seed=4)
    full = available_rows(side)
    kept = available_rows(side, raw_dir=raw, mate_keep=0.2)
    assert 0 < kept < full
    # The planned count is the number of rows the TRAIN stream yields.
    assert kept == len(load_sample(raw, side, TRAIN, 10**6, mate_keep=0.2))
    assert available_rows(side, raw_dir=raw, mate_keep=1.0) == full
    with pytest.raises(ValueError):
        available_rows(side, mate_keep=0.2)


def test_cache_path_changes_with_every_input():
    base = {"model_class": PstTaperedModel, "rep": Representation(mate_keep=0.10)}
    args = {"positions": 1000, "seed": 42, "quiet_only": False, "batch_size": 4096}
    reference = cache_path(**base, **args)

    assert cache_path(**base, **args) == reference
    assert cache_path(**base, **{**args, "positions": 2000}) != reference
    assert cache_path(**base, **{**args, "seed": 7}) != reference
    assert cache_path(**base, **{**args, "quiet_only": True}) != reference
    assert cache_path(**base, **{**args, "batch_size": 2048}) != reference
    assert (
        cache_path(model_class=PstTaperedModel, rep=Representation(mate_keep=0.5), **args)
        != reference
    )
    assert cache_path(model_class=PstTaperedModel, rep=Representation(k=0.01), **args) != reference
