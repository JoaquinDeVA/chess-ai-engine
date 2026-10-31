"""Tests of src/chess_lab/scripts/bench/compare_target_representations.py (the depth filter on a fixture shard)."""

import importlib.util
from pathlib import Path

from chess_lab.training.lichess_raw import shard_path, write_shard
from chess_lab.training.sidecar import build_sidecar
from tests.conftest import material_cp, random_positions
from tests.paths import ROOT


def load_script(name: str):
    path = next((ROOT / "src" / "chess_lab" / "scripts").glob(f"*/{name}.py"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bakeoff_fixture(raw: Path, side: Path, count: int = 1500) -> None:
    """One shard with mates (every ninth position) and mixed depths, and its sidecar."""
    raw.mkdir(parents=True)
    side.mkdir(parents=True)
    records = []
    for index, board in enumerate(random_positions(count, seed=8)):
        mate = 3 if index % 9 == 0 else None
        best = next(iter(board.legal_moves), None)
        records.append(
            {
                "fen": " ".join(board.fen().split()[:4]),
                "cp": None if mate else material_cp(board),
                "mate": mate,
                "depth": 12 + 8 * (index % 2),
                "knodes": 1,
                "best_move": best.uci() if best else None,
                "n_evals": 1,
                "n_pvs": 1,
            }
        )
    write_shard(shard_path(raw, 0), records, {"perspective": "white"})
    build_sidecar(shard_path(raw, 0), side)


def test_bakeoff_depth_filter_counts_kept_training_rows(tmp_path):
    module = load_script("compare_target_representations")
    raw, side = tmp_path / "raw", tmp_path / "side"
    bakeoff_fixture(raw, side)
    variant = next(v for v in module.VARIANTS if v.min_depth == 20)
    batches = list(module.training_batches(variant, raw, side, 100, 16, quiet=False))
    rows = sum(len(b) for b in batches)
    assert rows == 100
    assert all((b["depth"] >= 20).all() for b in batches)
