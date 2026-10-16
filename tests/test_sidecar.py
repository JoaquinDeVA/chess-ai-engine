"""Sidecar (split + quiet flag) and sampler over raw shards."""

import chess
import pandas as pd
import pyarrow.parquet as pq
import pytest

from chess_engine.training.dataset import TEST, TRAIN, VALIDATION, assign_split
from chess_engine.training.lichess_raw import read_metadata, shard_path, write_shard
from chess_engine.training.representation import is_quiet
from chess_engine.training.sidecar import build_sidecar, iter_batches, load_sample

START = chess.STARTING_FEN
CHECK = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"  # white in check
CAPTURE = "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"  # best move e4d5 captures


def record(fen, best_move="e2e4", cp=10):
    return {
        "fen": fen, "cp": cp, "mate": None, "depth": 20, "knodes": 1,
        "best_move": best_move, "n_evals": 1, "n_pvs": 1,
    }  # fmt: skip


def random_fens(count, offset=0):
    """Distinct legal positions: the start position after random-ish pawn/knight walks."""
    fens = []
    board = chess.Board()
    moves = list(board.legal_moves)
    for i in range(count):
        b = chess.Board()
        b.push(moves[(i + offset) % len(moves)])
        reply = list(b.legal_moves)
        b.push(reply[(i // len(moves)) % len(reply)])
        fens.append(b.fen())
    return list(dict.fromkeys(fens))


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr("chess_engine.training.lichess_raw.ROW_GROUP_SIZE", 64)  # many groups
    raw, side = tmp_path / "raw", tmp_path / "side"
    raw.mkdir()
    side.mkdir()
    fens = random_fens(400)
    for index, chunk in enumerate((fens[:250], fens[250:])):
        records = [record(fen, "e7e5" if " b " in fen else "e2e4") for fen in chunk]
        write_shard(shard_path(raw, index), records, {"perspective": "white"})
    for index in range(2):
        build_sidecar(shard_path(raw, index), side)
    return raw, side


def test_sidecar_aligns_with_shard(tmp_path):
    shard = shard_path(tmp_path, 0)
    records = [
        record(START),
        record(CHECK, "g1f3"),
        record(CAPTURE, "e4d5"),
        record("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1q"),
        record("4k3/8/8/8/8/8/8/4K3 b - - 0 1", None),
    ]
    write_shard(shard, records, {})
    out = build_sidecar(shard, tmp_path / "side")
    assert out == tmp_path / "side" / "part-00000.sidecar.parquet"
    table = pq.read_table(out)
    assert table.column_names == ["split", "quiet", "white_to_move"]
    assert table.num_rows == len(records)
    frame = table.to_pandas()
    fens = pd.Series([r["fen"] for r in records])
    assert frame["split"].tolist() == assign_split(fens).tolist()
    assert frame["quiet"].tolist() == [True, False, False, False, True]
    assert frame["white_to_move"].tolist() == [True, True, True, False, False]
    assert read_metadata(out) == {"rows": 5, "source": "part-00000.parquet"}


def test_load_sample_is_deterministic_and_respects_split(dirs):
    raw, side = dirs
    a = load_sample(raw, side, VALIDATION, 20, seed=1)
    b = load_sample(raw, side, VALIDATION, 20, seed=1)
    c = load_sample(raw, side, VALIDATION, 20, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a["fen"].equals(c["fen"])
    assert list(a.columns) == ["fen", "cp", "mate", "best_move", "quiet", "white_to_move"]
    assert len(a) == 20
    assert (assign_split(a["fen"]) == VALIDATION).all()
    assert a["fen"].is_unique
    for split in (TRAIN, TEST):
        assert (assign_split(load_sample(raw, side, split, 5)["fen"]) == split).all()


def test_load_sample_quiet_only(dirs):
    raw, side = dirs
    everything = load_sample(raw, side, TRAIN, 10_000)
    quiet = load_sample(raw, side, TRAIN, 10_000, quiet_only=True)
    assert quiet["quiet"].all() and len(quiet) <= len(everything)
    assert not everything["quiet"].all() or len(quiet) == len(everything)


def test_misaligned_sidecar_raises(dirs):
    raw, side = dirs
    path = side / "part-00001.sidecar.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(0, table.num_rows - 1), path)
    with pytest.raises(ValueError):
        load_sample(raw, side, TRAIN, 5)
    path.unlink()
    with pytest.raises(ValueError):
        load_sample(raw, side, TRAIN, 5)


def test_iter_batches_yields_n_rows_in_total(dirs):
    raw, side = dirs
    batches = list(iter_batches(raw, side, TRAIN, 16, 100, seed=3))
    assert sum(len(b) for b in batches) == 100
    assert all(len(b) <= 16 for b in batches)
    assert all((assign_split(b["fen"]) == TRAIN).all() for b in batches)
    again = list(iter_batches(raw, side, TRAIN, 16, 100, seed=3))
    pd.testing.assert_frame_equal(pd.concat(batches), pd.concat(again))
    everything = sum(len(b) for b in iter_batches(raw, side, TRAIN, 64, None, seed=3))
    assert everything == (assign_split(pd.Series(random_fens(400))) == TRAIN).sum()


def test_sampled_flags_match_own_fen(dirs):
    raw, side = dirs
    sample = load_sample(raw, side, TRAIN, 10_000)
    assert len(sample) > 100
    for row in sample.itertuples():
        board = chess.Board(row.fen)
        assert row.quiet == is_quiet(board, row.best_move)
        assert row.white_to_move == (board.turn == chess.WHITE)


def test_iter_batches_n_zero_yields_nothing(dirs):
    raw, side = dirs
    assert list(iter_batches(raw, side, TRAIN, 16, 0, seed=1)) == []
    assert all(len(b) for b in iter_batches(raw, side, TRAIN, 16, None, seed=1))
