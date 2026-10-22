"""Perspective detection, the pipeline script and the bake-off script."""

import importlib.util
import os
import subprocess
from pathlib import Path

import chess
import pytest

from chess_engine.training.lichess import SIDE_TO_MOVE, WHITE, detect_perspective
from chess_engine.training.lichess_raw import shard_path, write_shard
from chess_engine.training.sidecar import build_sidecar
from tests.conftest import material_cp, random_positions

ROOT = Path(__file__).resolve().parent.parent


def records_of(boards: list[chess.Board], perspective: str) -> list[dict]:
    """Fake records whose evaluation follows material, from the given perspective."""
    records = []
    for board in boards:
        cp = int(material_cp(board))
        if perspective == SIDE_TO_MOVE and board.turn == chess.BLACK:
            cp = -cp
        records.append({"fen": " ".join(board.fen().split()[:4]), "cp": cp, "mate": None})
    return records


@pytest.fixture(scope="module")
def boards():
    return random_positions(400, seed=21)


class TestPerspectiveDetection:
    @pytest.mark.parametrize("perspective", [WHITE, SIDE_TO_MOVE])
    def test_the_perspective_is_detected_from_the_data(self, boards, perspective):
        assert detect_perspective(records_of(boards, perspective)) == perspective

    def test_too_little_data_is_an_error(self, boards):
        with pytest.raises(ValueError, match="not enough"):
            detect_perspective(records_of(boards[:1], WHITE))


def load_script(name: str):
    path = next((ROOT / "scripts").glob(f"*/{name}.py"))
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


SCRIPT = ROOT / "scripts" / "run_full_pipeline.sh"
PIPELINE_MODELS = "linear_material linear_pst linear_king_bucket linear_pst_terms net4_pst"


def test_run_full_pipeline_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_run_full_pipeline_dry_run_prints_the_pipeline_and_writes_nothing(tmp_path):
    state = tmp_path / "state"
    env = {**os.environ, "DRY_RUN": "1", "STATE_DIR": str(state)}
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, check=True
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
    for model in PIPELINE_MODELS.split():
        assert any(
            "training.train" in line and f"--model {model} " in line and "--effort 30" in line
            for line in lines
        ), model
    assert any("comparison.evaluate --model h1 " in line for line in lines)
    assert any("--base 60 " in line and "--increment 1" in line for line in lines)
    assert any("--base 600 --increment 5" in line for line in lines)
    assert any("--max-depth 3" in line for line in lines)
    assert "mlp_features" not in result.stdout and "cnn_resnet" not in result.stdout
    assert not state.exists() or not list(state.glob("*.done"))
