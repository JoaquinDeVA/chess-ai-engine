"""Tests of the sparse-input neural evaluation core and of the ``net_pst_w64x16`` model."""

import json
import subprocess
import sys

import chess
import numpy as np
import pytest

from chess_lab.models import fit_stream, new_model
from chess_lab.models.definitions import (
    NeuralEvalModel,
    NnW64x16Model,
    get_model_class,
    load_evaluator,
)
from chess_lab.models.linear_features import features_of
from chess_lab.models.representation import Representation
from tests.conftest import mover_score, random_positions, white_score_of
from tests.lab.models.positions import frame


def random_net(seed: int = 0, rep: Representation | None = None) -> NeuralEvalModel:
    """A ``net_pst_w64x16`` with random layers, so that scores are not trivially 0.5."""
    model = new_model(NnW64x16Model, rep or Representation())
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


def test_net_has_the_documented_layer_shapes_and_parameter_count():
    model = new_model(NnW64x16Model, Representation())
    assert [weights.shape for weights, _ in model.layers] == [
        (768, 64),
        (64, 16),
        (16, 1),
    ]
    assert [bias.shape for _, bias in model.layers] == [(64,), (16,), (1,)]
    assert model.count_parameters() == 768 * 64 + 64 + 64 * 16 + 16 + 16 + 1
    assert NnW64x16Model.hidden == (64, 16)


def test_core_score_matches_a_dense_numpy_reference():
    model = random_net(seed=6)
    evaluator = model.evaluator()
    for board in random_positions(10, seed=8):
        assert evaluator.evaluate(board.fen(), board.turn) == pytest.approx(
            mover_score(dense_logit(model, board))
        )


def test_mirror_complement_holds_for_random_weights():
    model = random_net(seed=3)
    for board in random_positions(20, seed=11):
        assert white_score_of(model, board) == pytest.approx(
            1.0 - white_score_of(model, board.mirror())
        )


def test_fit_stream_rejects_empty_validation():
    model = new_model(NnW64x16Model, Representation())
    with pytest.raises(ValueError, match="validation set is empty"):
        fit_stream(model, [frame(random_positions(10, seed=1))], frame([]), {})


def test_save_load_round_trip_gives_identical_scores(tmp_path):
    rep = Representation(scale="cp_linear", perspective="white", filter="quiet")
    model = random_net(seed=4, rep=rep)
    path = model.save(tmp_path / "net_pst_w64x16.npz")

    loaded = get_model_class("net_pst_w64x16").load(path)

    assert Representation.from_dict(loaded.rep) == rep
    assert loaded.count_parameters() == model.count_parameters()
    for board in random_positions(10, seed=2):
        assert white_score_of(loaded, board) == pytest.approx(white_score_of(model, board))


def test_core_rejects_a_mismatching_input_layer(tmp_path):
    path = tmp_path / "wrong.npz"
    meta = {
        "features": "pst_slotted",
        "rep": Representation().to_dict(),
        "hidden": [64, 16],
    }
    arrays = {"W0": np.zeros((10, 64)), "b0": np.zeros(64)}
    for index, (weights, bias) in enumerate(
        new_model(NnW64x16Model, Representation()).layers[1:], start=1
    ):
        arrays[f"W{index}"], arrays[f"b{index}"] = weights, bias
    with open(path, "wb") as handle:
        np.savez(handle, meta=np.array(json.dumps(meta)), **arrays)
    with pytest.raises(ValueError, match="768"):
        load_evaluator("net_pst_w64x16", path)


def test_play_path_does_not_import_tensorflow(tmp_path):
    path = random_net(seed=7).save(tmp_path / "net_pst_w64x16.npz")
    script = (
        "import sys, chess\n"
        "from chess_lab.models.definitions import load_evaluator\n"
        f"evaluator = load_evaluator('net_pst_w64x16', {str(path)!r})\n"
        "score = evaluator.evaluate(chess.Board().fen(), True)\n"
        "print(0.0 < score < 1.0, 'tensorflow' in sys.modules, 'keras' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.split() == ["True", "False", "False"]
