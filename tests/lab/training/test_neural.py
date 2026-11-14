"""Tests of the sparse-input neural evaluation core and of the ``net_pst_w64x16`` model."""

import json
import subprocess
import sys

import chess
import numpy as np
import pytest

from chess_lab.models.definitions import (
    NeuralEvalModel,
    NnK4ThreatsW64BothModel,
    NnK4W64BothModel,
    NnThreatsW64BothModel,
    NnW64BothModel,
    NnW64x16Model,
    get_model_class,
    load_evaluator,
)
from chess_lab.models.features import features_of
from chess_lab.models.target import Representation
from chess_lab.training.neural import fit_stream, new_model
from tests.conftest import mover_score, random_positions, white_score_of
from tests.lab.models.positions import frame


def random_net(
    seed: int = 0, rep: Representation | None = None, model_class: type = NnW64x16Model
) -> NeuralEvalModel:
    """A network with random layers, so that scores are not trivially 0.5."""
    model = new_model(model_class, rep or Representation())
    rng = np.random.default_rng(seed)
    model.layers = [
        (rng.normal(0.0, 0.3, size=weights.shape), rng.normal(0.0, 0.1, size=bias.shape))
        for weights, bias in model.layers
    ]
    return model


def dense_logit(model: NeuralEvalModel, board: chess.Board) -> float:
    """Reference forward pass on a dense 768- (or 769-) vector, same weights as ``model``."""
    indices, values = features_of(model).active(board)
    x = np.zeros(features_of(model).size())
    for index, value in zip(indices, values, strict=True):
        x[index] += value
    (w0, b0), *rest = model.layers
    h = np.clip(x @ w0 + b0, 0.0, 1.0)
    if model.perspective:
        # the rival's view: every piece in the other block of its square, in the bucket of the
        # rival's king; the dense entries have no rival view
        features = features_of(model)
        king = board.king(not board.turn)
        if board.turn == chess.WHITE:
            king = chess.square_mirror(king)
        pieces = getattr(features, "buckets", 1) * 768
        shift = features.bucket(king) * 768 if hasattr(features, "bucket") else 0
        rival = np.zeros_like(x)
        for index, value in zip(indices, values, strict=True):
            if index < pieces:
                rival[shift + (index % 768 + 384) % 768] += value
        h = np.concatenate([h, np.clip(rival @ w0 + b0, 0.0, 1.0)])
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


def test_both_perspectives_double_the_input_of_the_second_layer():
    model = new_model(NnW64BothModel, Representation())
    assert [weights.shape for weights, _ in model.layers] == [(768, 64), (128, 1)]
    assert model.perspective and not NnW64x16Model.perspective
    assert model.count_parameters() == 768 * 64 + 64 + 128 + 1


BOTH_MODELS = [NnW64BothModel, NnThreatsW64BothModel, NnK4W64BothModel, NnK4ThreatsW64BothModel]


@pytest.mark.parametrize("model_class", [NnW64x16Model, *BOTH_MODELS])
def test_core_score_matches_a_dense_numpy_reference(model_class):
    model = random_net(seed=6, model_class=model_class)
    evaluator = model.evaluator()
    for board in random_positions(10, seed=8):
        assert evaluator.evaluate(board.fen(), board.turn) == pytest.approx(
            mover_score(dense_logit(model, board))
        )


@pytest.mark.parametrize("model_class", BOTH_MODELS)
def test_the_training_graph_computes_what_the_core_does(model_class):
    from chess_lab.training.epochs import encode_frame
    from chess_lab.training.neural import _assign, _keras_network, _padded, _training_keras

    model = random_net(seed=5, model_class=model_class)
    features = features_of(model)
    width = features.max_entries()
    keras = _training_keras()
    network, parts = _keras_network(
        keras, [w.shape for w, _ in model.layers], width, 0.0, True, getattr(features, "buckets", 1)
    )
    _assign(parts, model.layers)
    boards = random_positions(40, seed=9)
    index, value, _ = _padded(encode_frame(features, Representation(), frame(boards)), width)
    logits = np.asarray(network.predict([index, value], verbose=0)).reshape(-1)
    for board, logit in zip(boards, logits, strict=True):
        assert logit == pytest.approx(dense_logit(model, board), abs=1e-4)


@pytest.mark.parametrize("model_class", [NnW64x16Model, NnW64BothModel])
def test_mirror_complement_holds_for_random_weights(model_class):
    model = random_net(seed=3, model_class=model_class)
    for board in random_positions(20, seed=11):
        assert white_score_of(model, board) == pytest.approx(
            1.0 - white_score_of(model, board.mirror())
        )


def test_fit_stream_rejects_empty_validation():
    model = new_model(NnW64x16Model, Representation())
    with pytest.raises(ValueError, match="validation set is empty"):
        fit_stream(model, [frame(random_positions(10, seed=1))], frame([]), {})


def test_save_load_round_trip_gives_identical_scores(tmp_path):
    rep = Representation(scale="cp_linear")
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
