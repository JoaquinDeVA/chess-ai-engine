"""Tests of the sparse-input neural evaluation core and of the ``net4_pst`` model."""

import json
import subprocess
import sys
import time

import chess
import numpy as np
import pandas as pd
import pytest

from chess_engine.models.native import model_evaluator
from chess_engine.models.neural_eval import NeuralEvalModel, Nn4Model
from chess_engine.models.registry import get_model_class
from chess_lab.models import fit_stream, new_model
from chess_lab.models.linear_features import features_of
from chess_lab.models.python_eval import PythonEvaluator
from chess_lab.models.representation import Representation
from tests.conftest import material_cp, random_positions

WHITE_UP_A_QUEEN = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_UP_A_QUEEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"


def frame(boards: list[chess.Board]) -> pd.DataFrame:
    """Raw-shard-like rows whose White-POV centipawns are the material balance."""
    return pd.DataFrame(
        {
            "fen": [board.fen() for board in boards],
            "cp": [material_cp(board) for board in boards],
            "mate": np.full(len(boards), np.nan),
            "best_move": [None] * len(boards),
            "quiet": [True] * len(boards),
            "white_to_move": [board.turn == chess.WHITE for board in boards],
        }
    )


def random_nn4(seed: int = 0, rep: Representation | None = None) -> NeuralEvalModel:
    """A ``net4_pst`` with random layers, so that scores are not trivially 0.5."""
    model = new_model(Nn4Model, rep or Representation())
    rng = np.random.default_rng(seed)
    model.layers = [
        (rng.normal(0.0, 0.3, size=weights.shape), rng.normal(0.0, 0.1, size=bias.shape))
        for weights, bias in model.layers
    ]
    return model


def dense_logit(model: NeuralEvalModel, board: chess.Board) -> float:
    """Reference forward pass on a dense 768- (or 769-) vector, same weights as ``model``."""
    indices, values = features_of(model).active(board, model.perspective)
    x = np.zeros(features_of(model).size(model.perspective))
    for index, value in zip(indices, values, strict=True):
        x[index] += value
    (w0, b0), *rest = model.layers
    h = np.clip(x @ w0 + b0, 0.0, 1.0)
    for weights, bias in rest[:-1]:
        h = np.maximum(h @ weights + bias, 0.0)
    last_weights, last_bias = rest[-1]
    return float((h @ last_weights + last_bias)[0])


def test_nn4_has_the_documented_layer_shapes_and_parameter_count():
    model = new_model(Nn4Model, Representation())
    assert [weights.shape for weights, _ in model.layers] == [
        (768, 256),
        (256, 32),
        (32, 32),
        (32, 1),
    ]
    assert [bias.shape for _, bias in model.layers] == [(256,), (32,), (32,), (1,)]
    assert model.count_parameters() == 206_177
    assert Nn4Model.hidden == (256, 32, 32)


def test_logit_matches_a_dense_numpy_reference():
    model = random_nn4(seed=6)
    for board in random_positions(10, seed=8):
        assert PythonEvaluator(model).logit(board) == pytest.approx(dense_logit(model, board))


def test_mirror_complement_holds_for_random_weights():
    model = random_nn4(seed=3)
    for board in random_positions(20, seed=11):
        assert PythonEvaluator(model).expected_score(board) == pytest.approx(
            1.0 - PythonEvaluator(model).expected_score(board.mirror())
        )


def test_fit_stream_rejects_empty_validation():
    model = new_model(Nn4Model, Representation())
    with pytest.raises(ValueError, match="validation set is empty"):
        fit_stream(model, [frame(random_positions(10, seed=1))], frame([]), {})


def test_save_load_round_trip_gives_identical_scores(tmp_path):
    rep = Representation(scale="cp_linear", perspective="white", filter="quiet")
    model = random_nn4(seed=4, rep=rep)
    path = model.save(tmp_path / "net4_pst.npz")

    loaded = get_model_class("net4_pst").load(path)

    assert Representation.from_dict(loaded.rep) == rep
    assert loaded.count_parameters() == model.count_parameters()
    for board in random_positions(10, seed=2):
        assert PythonEvaluator(loaded).expected_score(board) == pytest.approx(
            PythonEvaluator(model).expected_score(board)
        )
        assert PythonEvaluator(loaded).predict_cp(board) == pytest.approx(
            PythonEvaluator(model).predict_cp(board)
        )


def test_core_rejects_a_mismatching_input_layer(tmp_path):
    path = tmp_path / "wrong.npz"
    meta = {
        "features": "pst_slotted",
        "rep": Representation().to_dict(),
        "hidden": [256, 32, 32],
    }
    arrays = {"W0": np.zeros((10, 256)), "b0": np.zeros(256)}
    for index, (weights, bias) in enumerate(
        new_model(Nn4Model, Representation()).layers[1:], start=1
    ):
        arrays[f"W{index}"], arrays[f"b{index}"] = weights, bias
    with open(path, "wb") as handle:
        np.savez(handle, meta=np.array(json.dumps(meta)), **arrays)
    model = get_model_class("net4_pst").load(path)
    with pytest.raises(ValueError, match="768"):
        model_evaluator(model)


def test_play_path_does_not_import_tensorflow(tmp_path):
    path = random_nn4(seed=7).save(tmp_path / "net4_pst.npz")
    script = (
        "import sys, chess\n"
        "from chess_engine.models.registry import get_model_class\n"
        "from chess_engine.models.native import model_evaluator\n"
        f"model = get_model_class('net4_pst').load({str(path)!r})\n"
        "score = model_evaluator(model).evaluate(chess.Board().fen(), True)\n"
        "print(0.0 < score < 1.0, 'tensorflow' in sys.modules, 'keras' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.split() == ["True", "False", "False"]


def test_logit_cost_is_reported():
    model = random_nn4(seed=9)
    boards = random_positions(200, seed=10)
    started = time.perf_counter()
    for board in boards:
        PythonEvaluator(model).logit(board)
    micros = (time.perf_counter() - started) / len(boards) * 1e6
    print(f"\nnn4 logit: {micros:.1f} us per call")
