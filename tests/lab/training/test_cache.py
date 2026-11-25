"""Directory of the encoded-batch cache and the frames in it."""

import os

import chess
import numpy as np
import pytest

from chess_lab.datasets.raw_shards import shard_path, write_shard
from chess_lab.datasets.sidecar import build_sidecar
from chess_lab.models.definitions import NnThreatsW64BothModel, NnW64BothModel, NnW128BothModel
from chess_lab.models.target import Representation
from chess_lab.training import cache
from chess_lab.training.cache import cache_path, load_frame, save_frame


def record(fen):
    return {
        "fen": fen, "cp": 10, "mate": None, "depth": 20, "knodes": 1,
        "best_move": None, "n_evals": 1, "n_pvs": 1,
    }  # fmt: skip


def make_shard(raw, side, index, count):
    fens = []
    board = chess.Board()
    for move in list(board.legal_moves)[:count]:
        board.push(move)
        fens.append(board.fen())
        board.pop()
    write_shard(shard_path(raw, index), [record(fen) for fen in fens], {"perspective": "white"})
    build_sidecar(shard_path(raw, index), side)


@pytest.fixture
def data(tmp_path):
    raw, side = tmp_path / "raw", tmp_path / "side"
    raw.mkdir()
    make_shard(raw, side, 0, 3)
    return raw, side


def test_cache_path_changes_with_every_input(data):
    base = {"model_class": NnW64BothModel, "rep": Representation(mate_keep=0.10)}
    args = {"positions": 1000, "seed": 42, "quiet_only": False, "batch_size": 4096}
    args["raw_dir"], args["sidecar_dir"] = data
    reference = cache_path(**base, **args)

    assert cache_path(**base, **args) == reference
    assert cache_path(**base, **{**args, "positions": 2000}) != reference
    assert cache_path(**base, **{**args, "seed": 7}) != reference
    assert cache_path(**base, **{**args, "quiet_only": True}) != reference
    assert cache_path(**base, **{**args, "batch_size": 2048}) != reference
    assert (
        cache_path(model_class=NnW64BothModel, rep=Representation(mate_keep=0.5), **args)
        != reference
    )
    assert cache_path(model_class=NnW64BothModel, rep=Representation(k=0.01), **args) != reference


def test_cache_path_depends_on_feature_set_not_on_class(data):
    args = {"rep": Representation(), "positions": 1000, "seed": 42, "quiet_only": True}
    args["batch_size"] = 4096
    args["raw_dir"], args["sidecar_dir"] = data

    assert cache_path(model_class=NnW128BothModel, **args) == cache_path(
        model_class=NnW64BothModel, **args
    )
    assert cache_path(model_class=NnThreatsW64BothModel, **args) != cache_path(
        model_class=NnW64BothModel, **args
    )


def test_cache_path_follows_the_data_and_the_split(data, monkeypatch):
    raw, side = data
    args = {"model_class": NnW64BothModel, "rep": Representation(), "positions": 1000}
    args |= {"seed": 42, "quiet_only": False, "batch_size": 4096}
    reference = cache_path(raw_dir=raw, sidecar_dir=side, **args)

    assert cache_path(raw_dir=raw, sidecar_dir=side, **args) == reference
    monkeypatch.setattr(cache, "SPLIT_SEED", 43)
    assert cache_path(raw_dir=raw, sidecar_dir=side, **args) != reference
    monkeypatch.undo()
    monkeypatch.setattr(cache, "DEFAULT_TEST_FRACTION", 0.2)
    assert cache_path(raw_dir=raw, sidecar_dir=side, **args) != reference
    monkeypatch.undo()
    make_shard(raw, side, 1, 2)
    assert cache_path(raw_dir=raw, sidecar_dir=side, **args) != reference


def test_cache_path_changes_when_a_shard_is_rewritten(data):
    raw, side = data
    args = {"model_class": NnW64BothModel, "rep": Representation(), "positions": 1000}
    args |= {"seed": 42, "quiet_only": False, "batch_size": 4096}
    reference = cache_path(raw_dir=raw, sidecar_dir=side, **args)
    make_shard(raw, side, 0, 5)  # same name, other rows
    assert cache_path(raw_dir=raw, sidecar_dir=side, **args) != reference


def test_a_cut_between_writing_and_renaming_an_encoded_frame_leaves_no_frame(tmp_path, monkeypatch):
    encoded = (np.zeros(1, int), np.zeros(1, int), np.ones(1), np.ones(1))

    def cut(*_):
        raise OSError("cut")

    monkeypatch.setattr(os, "replace", cut)
    with pytest.raises(OSError):
        save_frame(tmp_path, 0, encoded)
    (tmp_path / "0000001.tmp").write_bytes(b"half a file")
    assert load_frame(tmp_path, 0) is None
    assert load_frame(tmp_path, 1) is None


def test_a_saved_frame_is_read_back(tmp_path):
    encoded = (np.arange(3), np.arange(3) + 5, np.ones(3), np.array([0.25, 0.5]))
    save_frame(tmp_path, 4, encoded)
    assert (tmp_path / "0000004.npz").exists()
    for got, want in zip(load_frame(tmp_path, 4), encoded, strict=True):
        np.testing.assert_array_equal(got, want)
    assert load_frame(None, 4) is None
