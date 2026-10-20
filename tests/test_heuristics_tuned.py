"""Tests of the classical reference ``h1`` and of the script that fits its weights."""

import importlib.util
import json
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_engine.bots.iterative_heuristic_bot import IterativeHeuristicBot
from chess_engine.comparison import tournament
from chess_engine.comparison.tradeoff import (
    HEURISTICS,
    fixed_time_manager,
    make_player,
    validate_specs,
)
from chess_engine.evaluation import heuristics
from chess_engine.evaluation.heuristics import (
    FEATURE_NAMES,
    h1,
    load_h1_weights,
    reset_h1_cache,
)
from chess_engine.search.chess_game_state import ChessGameState
from chess_engine.training.lichess_raw import shard_path, write_shard
from chess_engine.training.sidecar import build_sidecar
from tests.conftest import random_positions

ROOT = Path(__file__).resolve().parent.parent
TUNED_WEIGHTS = {
    "material_diff_normalized": 0.6,
    "positional_diff_normalized": 0.1,
    "mobility_diff_normalized": 0.3,
    "center_diff_normalized": 0.05,
    "king_safety_diff_normalized": -0.2,
    "development_diff_normalized": 0.15,
}


def load_script(relative: str, name: str):
    """Import a script from ``scripts/`` as a module (the scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fit_script():
    return load_script("scripts/data/fit_h1_weights.py", "fit_h1_weights")


@pytest.fixture(autouse=True)
def weights_file(tmp_path, monkeypatch):
    """Point ``h1`` at a weights file in a temporary directory, with a fresh cache."""
    path = tmp_path / "h1_weights.json"
    path.write_text(json.dumps({"weights": TUNED_WEIGHTS}))
    monkeypatch.setattr(heuristics, "H1_WEIGHTS_PATH", path)
    reset_h1_cache()
    yield path
    reset_h1_cache()


def states(count, seed):
    return [ChessGameState(board) for board in random_positions(count, seed=seed, max_plies=40)]


def test_h1_is_complementary_for_white_and_black():
    for state in states(50, seed=3):
        white = h1(state, chess.WHITE)
        black = h1(state, chess.BLACK)
        assert 0.01 <= white <= 0.99
        assert white + black == pytest.approx(1.0)


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


def test_missing_weights_file_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/data/fit_h1_weights.py"):
        load_h1_weights(tmp_path / "absent.json")


def test_weights_are_cached_until_reset(weights_file):
    assert load_h1_weights() == TUNED_WEIGHTS
    other = {**TUNED_WEIGHTS, "material_diff_normalized": 0.9}
    weights_file.write_text(json.dumps({"weights": other}))
    assert load_h1_weights() == TUNED_WEIGHTS  # served from the cache
    reset_h1_cache()
    assert load_h1_weights() == other


def test_shipped_weights_file_loads_with_the_feature_names():
    reset_h1_cache()
    weights = load_h1_weights(Path(heuristics.__file__).with_name("h1_weights.json"))
    assert set(weights) == set(FEATURE_NAMES)
    assert all(np.isfinite(value) for value in weights.values())


def test_tradeoff_accepts_h1_spec():
    assert HEURISTICS["h1"] is h1
    validate_specs(["h1"])
    player = make_player("h1", fixed_time_manager(0.2))
    assert isinstance(player, IterativeHeuristicBot)
    assert player.eval_function is h1


def test_play_tournament_defaults_use_h1_anchor():
    script = load_script("scripts/bench/play_tournament.py", "play_tournament_h1")
    assert tournament.ANCHOR == "h1"
    args = script.parse_args([])
    assert args.anchor == "h1"
    assert args.specs == ["linear_pst", "linear_pst_terms", "linear_king_bucket", "net4_pst", "h1"]
