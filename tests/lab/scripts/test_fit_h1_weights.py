"""Tests of src/chess_lab/scripts/data/fit_h1_weights.py: the fit and the command on fixture shards."""

import importlib.util
import json

import numpy as np
import pytest

from chess_engine.evaluation.heuristics import FEATURE_NAMES
from chess_lab.training.lichess_raw import shard_path, write_shard
from chess_lab.training.sidecar import build_sidecar
from tests.conftest import random_positions
from tests.paths import ROOT


def load_script(relative: str, name: str):
    """Import a script from ``scripts/`` as a module (the scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fit_script():
    return load_script("src/chess_lab/scripts/data/fit_h1_weights.py", "fit_h1_weights")


def test_fit_recovers_known_weights(fit_script):
    rng = np.random.default_rng(7)
    features = rng.uniform(0.0, 1.0, size=(2000, len(FEATURE_NAMES)))
    true = np.array([0.3, -0.2, 0.5, 0.1, 0.0, 0.4])
    targets = 0.5 + (features - 0.5) @ true + rng.normal(0.0, 0.01, size=2000)
    fitted = fit_script.fit_weights(features, targets)
    assert np.allclose(fitted, true, atol=0.05)


def test_fit_script_runs_on_fixture_shards(fit_script, tmp_path, capsys):
    raw, sidecar = tmp_path / "raw", tmp_path / "sidecar"
    raw.mkdir()
    sidecar.mkdir()
    boards = random_positions(400, seed=11, max_plies=60)
    fens = [board.fen() for board in boards]
    for index, chunk in enumerate((fens[:200], fens[200:])):
        records = [
            {
                "fen": fen,
                "cp": float((i * 37) % 800 - 400),
                "mate": None,
                "depth": 20,
                "knodes": 1,
                "best_move": None,
                "n_evals": 1,
                "n_pvs": 1,
            }
            for i, fen in enumerate(chunk)
        ]
        write_shard(shard_path(raw, index), records, {"perspective": "white"})
        build_sidecar(shard_path(raw, index), sidecar)
    out = tmp_path / "h1_weights.json"

    argv = ["--raw", str(raw), "--sidecar", str(sidecar), "--positions", "300"]
    code = fit_script.main([*argv, "--jobs", "1", "--out", str(out)])

    assert code == 0
    payload = json.loads(out.read_text())
    assert set(payload["weights"]) == set(FEATURE_NAMES)
    assert np.isfinite(payload["r2"]) and np.isfinite(payload["r2_h3"])
    assert payload["n"] > 0
    assert "held-out R2" in capsys.readouterr().out
