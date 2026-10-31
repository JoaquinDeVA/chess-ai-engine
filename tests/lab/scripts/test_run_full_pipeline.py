"""Tests of src/chess_lab/scripts/run_full_pipeline.sh: valid bash and a dry run."""

import os
import subprocess

from tests.paths import ROOT

SCRIPT = ROOT / "src" / "chess_lab" / "scripts" / "run_full_pipeline.sh"
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
    assert any("measure.evaluate --model h1 " in line for line in lines)
    assert "tournament" not in result.stdout
    assert "mlp_features" not in result.stdout and "cnn_resnet" not in result.stdout
    assert not state.exists() or not list(state.glob("*.done"))
