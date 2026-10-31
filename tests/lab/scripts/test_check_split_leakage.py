"""Tests of src/chess_lab/scripts/data/check_split_leakage.py with a handful of positions."""

import importlib.util

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chess_lab.training.dataset import TEST, TRAIN, VALIDATION
from tests.paths import ROOT


@pytest.fixture(scope="module")
def script():
    path = ROOT / "src" / "chess_lab" / "scripts" / "data" / "check_split_leakage.py"
    spec = importlib.util.spec_from_file_location("check_split_leakage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def key(*ucis):
    board = chess.Board()
    for uci in ucis:
        board.push_uci(uci)
    return " ".join(board.fen().split()[:4])


def write_shard(tmp_path, rows):
    """One shard ``part-00000`` and its sidecar from ``[(fen key, split)]``."""
    raw, side = tmp_path / "raw", tmp_path / "side"
    raw.mkdir(exist_ok=True)
    side.mkdir(exist_ok=True)
    pq.write_table(pa.table({"fen": [k for k, _ in rows]}), raw / "part-00000.parquet")
    pq.write_table(
        pa.table({"split": pa.array([s for _, s in rows], pa.int8())}),
        side / "part-00000.sidecar.parquet",
    )
    return raw, side


def test_a_position_repeated_inside_one_split_is_not_a_leak(script, tmp_path):
    raw, side = write_shard(tmp_path, [(key(), TRAIN), (key("e2e4"), TRAIN), (key(), TRAIN)])
    hashes, splits = script.load_index(raw, side)
    assert script.duplicate_report(hashes, splits) == {
        "rows": 3,
        "repeated_rows": 2,
        "cross_split_rows": 0,
    }


def test_a_position_in_two_splits_is_reported(script, tmp_path):
    raw, side = write_shard(tmp_path, [(key(), TRAIN), (key(), TEST), (key("e2e4"), VALIDATION)])
    hashes, splits = script.load_index(raw, side)
    report = script.duplicate_report(hashes, splits)
    assert report["cross_split_rows"] == 2 and report["repeated_rows"] == 2


def test_neighbours_and_mirror_images_of_a_test_position(script, tmp_path):
    mirrored_start = " ".join(chess.Board().mirror().fen().split()[:4])
    raw, side = write_shard(
        tmp_path,
        [(key(), TEST), (key("e2e4"), TRAIN), (mirrored_start, TRAIN), (key("d2d4"), VALIDATION)],
    )
    hashes, splits = script.load_index(raw, side)
    report = script.neighbour_report(hashes, splits, [chess.STARTING_FEN])
    assert report["positions"] == 1
    assert report["mirror_in_data"] == 1.0 and report["mirror_in_train"] == 1.0
    assert report["child_in_data"] == 1.0 and report["child_in_train"] == 1.0


def test_a_test_position_without_relatives_reports_zero(script, tmp_path):
    raw, side = write_shard(tmp_path, [(key(), TEST), (key("e2e4", "e7e5"), TRAIN)])
    hashes, splits = script.load_index(raw, side)
    report = script.neighbour_report(hashes, splits, [chess.STARTING_FEN])
    assert report["child_in_data"] == 0.0 and report["mirror_in_data"] == 0.0


def test_splits_of_returns_minus_one_for_unknown_keys(script):
    keys = ["a", "b"]
    order = sorted(keys, key=hash)
    hashes = np.array([hash(k) for k in order], dtype=np.int64)
    splits = np.array([TRAIN, TEST], dtype=np.int8)
    result = script.splits_of(hashes, splits, [order[1], "unknown"])
    assert result == [TEST, -1]


def test_sample_test_fens_returns_only_test_rows(script, tmp_path):
    rows = [(key(), TRAIN), (key("e2e4"), TEST), (key("d2d4"), TEST), (key("c2c4"), VALIDATION)]
    raw, side = write_shard(tmp_path, rows)
    fens = script.sample_test_fens(raw, side, count=10, seed=1)
    assert sorted(fens) == sorted([key("e2e4"), key("d2d4")])


def test_has_train_child_flags_only_positions_with_a_child_in_train(script, tmp_path):
    after_e4 = chess.Board()
    after_e4.push_uci("e2e4")
    raw, side = write_shard(
        tmp_path,
        [
            (key(), TEST),
            (key("e2e4"), TRAIN),
            (key("e2e4", "e7e5"), VALIDATION),
            (key("d2d4"), TEST),
        ],
    )
    hashes, splits = script.load_index(raw, side)
    flags = script.has_train_child(
        hashes, splits, [chess.STARTING_FEN, after_e4.fen(), chess.Board().fen()]
    )
    # the start has the child e4 in train; after 1.e4 the only child in the data is in validation
    assert flags.tolist() == [True, False, True]
