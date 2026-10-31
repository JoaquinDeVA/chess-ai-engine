"""Raw Parquet export of the Lichess evaluation database."""

import json

import pyarrow.parquet as pq
import pytest

from chess_lab.training.lichess_raw import (
    RAW_COLUMNS,
    is_legal_position,
    parse_raw_record,
    read_metadata,
    shard_path,
    write_shard,
)
from tests.paths import ROOT

FEN = "8/4r3/2R2pk1/6pp/3P4/6P1/5K1P/8 b - -"


def line(evals) -> str:
    return json.dumps({"fen": FEN, "evals": evals})


class TestParseRawRecord:
    def test_cp_is_kept_as_stored(self):
        pvs = [{"cp": -31, "line": "e7a7 f2e3"}, {"cp": 5, "line": "g6f5"}]
        record = parse_raw_record(line([{"pvs": pvs, "knodes": 10, "depth": 30}]))
        assert record == {
            "fen": FEN, "cp": -31, "mate": None, "depth": 30, "knodes": 10,
            "best_move": "e7a7", "n_evals": 1, "n_pvs": 2,
        }  # fmt: skip

    def test_mate(self):
        record = parse_raw_record(line([{"pvs": [{"mate": 3, "line": "a1a2"}], "depth": 9}]))
        assert record["mate"] == 3 and record["cp"] is None

    def test_deepest_then_most_nodes(self):
        evals = [
            {"pvs": [{"cp": 1, "line": "a"}], "knodes": 5, "depth": 20},
            {"pvs": [{"cp": 2, "line": "b"}], "knodes": 9, "depth": 20},
            {"pvs": [{"cp": 3, "line": "c"}], "knodes": 99, "depth": 10},
        ]
        record = parse_raw_record(line(evals))
        assert (record["cp"], record["n_evals"]) == (2, 3)

    def test_missing_line_gives_no_best_move(self):
        record = parse_raw_record(line([{"pvs": [{"cp": 0}], "knodes": 1, "depth": 1}]))
        assert record["best_move"] is None

    @pytest.mark.parametrize(
        "text",
        ["not json", "{}", '{"fen": "x", "evals": []}', line([{"pvs": [], "depth": 1}]), "[1]"],
    )
    def test_unusable_lines(self, text):
        assert parse_raw_record(text) is None


ILLEGAL_FEN = "8/8/8/8/8/8/8/KK5k w - -"


@pytest.mark.parametrize(
    "fen, legal",
    [
        (FEN, True),
        (ILLEGAL_FEN, False),
        ("garbage", False),
        ("rnbqkbnr/8/8/8/8/8/8/8 w - -", False),
    ],
)
def test_is_legal_position(fen, legal):
    assert is_legal_position(fen) is legal


def test_shard_roundtrip(tmp_path):
    records = [
        parse_raw_record(
            line([{"pvs": [{"cp": -7000, "line": "e7a7"}], "knodes": 3, "depth": 40}])
        ),
        parse_raw_record(line([{"pvs": [{"mate": -2}], "knodes": 1, "depth": 12}])),
    ]
    path = shard_path(tmp_path, 0)
    write_shard(path, records, {"perspective": "white", "lines": 2})
    table = pq.read_table(path)
    assert table.column_names == list(RAW_COLUMNS)
    assert table.to_pylist() == records
    assert read_metadata(path) == {"perspective": "white", "lines": 2}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.slow  # starts a pool of worker processes
def test_fetch_is_sharded_and_restartable(tmp_path):
    import importlib.util

    import zstandard

    from tests.conftest import material_cp, random_positions

    spec = importlib.util.spec_from_file_location(
        "build_raw_shards",
        ROOT / "src/chess_lab/scripts/data/build_raw_shards.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    lines = []
    for board in random_positions(400, seed=1):
        cp = int(material_cp(board))
        fen = " ".join(board.fen().split()[:4])
        pv = {"cp": cp, "line": "e2e4"}
        lines.append(json.dumps({"fen": fen, "evals": [{"pvs": [pv], "knodes": 1, "depth": 5}]}))
    lines.insert(10, "garbage")
    illegal = json.dumps(
        {"fen": ILLEGAL_FEN, "evals": [{"pvs": [{"cp": 0, "line": "a1a2"}], "depth": 5}]}
    )
    lines.insert(20, illegal)
    source = tmp_path / "db.jsonl.zst"
    source.write_bytes(zstandard.ZstdCompressor().compress(("\n".join(lines) + "\n").encode()))
    log = tmp_path / "progress.log"

    out = tmp_path / "raw"
    first = module.fetch(str(source), 300, out, 150, workers=2, log_file=log)
    assert (first["lines"], first["kept"], first["dropped"], first["illegal"]) == (300, 298, 1, 1)
    assert sorted(p.name for p in out.glob("*.parquet")) == [
        f"part-0000{i}.parquet" for i in range(2)
    ]
    assert sum(pq.read_metadata(p).num_rows for p in out.glob("*.parquet")) == 298
    assert "ETA" in log.read_text() and "100.0%" in log.read_text()
    mtime = shard_path(out, 0).stat().st_mtime_ns

    shard_path(out, 1).unlink()
    again = module.fetch(source.as_uri(), 300, out, 150)
    assert (again["kept"], again["dropped"], again["illegal"]) == (298, 1, 1)
    assert shard_path(out, 0).stat().st_mtime_ns == mtime
    assert read_metadata(shard_path(out, 0))["perspective"] == again["perspective"]

    whole = module.fetch(str(source), None, tmp_path / "all", 150, workers=2)
    assert (whole["lines"], whole["kept"], whole["dropped"], whole["illegal"]) == (402, 400, 1, 1)
    assert len(list((tmp_path / "all").glob("*.parquet"))) == 3
