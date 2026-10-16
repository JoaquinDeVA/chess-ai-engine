"""Effort-to-configuration curves and the sidecar row count they depend on."""

import pandas as pd
import pytest

from chess_engine.training.dataset import TRAIN, assign_split
from chess_engine.training.effort import available_rows, effort_config
from chess_engine.training.lichess_raw import shard_path, write_shard
from chess_engine.training.sidecar import build_sidecar
from tests.conftest import random_positions


def test_effort_bounds():
    assert effort_config(0, 10**9)["positions"] == 100_000
    assert effort_config(100, 10**9)["positions"] == 10**9


def test_effort_is_monotonic():
    previous = None
    for effort in range(101):
        config = effort_config(effort, 10**9)
        if previous is not None:
            for key in ("positions", "max_epochs", "trials"):
                assert config[key] >= previous[key], (effort, key)
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


@pytest.mark.parametrize("effort", [-1, 101])
def test_effort_rejects_out_of_range(effort):
    with pytest.raises(ValueError):
        effort_config(effort, 10**6)


def test_available_rows_counts_train_rows_from_sidecars(tmp_path):
    raw, side = tmp_path / "raw", tmp_path / "side"
    raw.mkdir()
    fens = [b.fen() for b in random_positions(120, seed=5)]
    write_shard(
        shard_path(raw, 0),
        [
            {"fen": f, "cp": 0, "mate": None, "depth": 20, "knodes": 1, "best_move": None,
             "n_evals": 1, "n_pvs": 1}  # fmt: skip
            for f in fens
        ],
        {},
    )  # fmt: skip
    build_sidecar(shard_path(raw, 0), side)
    expected = int((assign_split(pd.Series(fens)) == TRAIN).sum())
    assert available_rows(side) == expected
    assert available_rows(side, quiet_only=True) <= expected
    assert available_rows(tmp_path / "empty") == 0
