"""Tests of the sparse-input neural evaluation core."""

import json
import subprocess
import sys

import chess
import numpy as np
import pytest

from chess_lab.models.definitions import (
    NeuralEvalModel,
    NnThreatsW64BothModel,
    NnW64BothModel,
    get_model_class,
)
from chess_lab.models.features import (
    KING_BASE,
    KING_BUCKETS,
    MATERIAL_BASE,
    PAWN_FILES_BASE,
    features_of,
)
from chess_lab.models.target import Representation
from chess_lab.training.neural import fit_stream, new_model
from tests.conftest import mover_score, random_positions, white_score_of
from tests.lab.models.positions import frame


def random_net(
    seed: int = 0, rep: Representation | None = None, model_class: type = NnW64BothModel
) -> NeuralEvalModel:
    """A network with random layers, so that scores are not trivially 0.5."""
    model = new_model(model_class, rep or Representation())
    rng = np.random.default_rng(seed)
    model.layers = [
        (rng.normal(0.0, 0.3, size=weights.shape), rng.normal(0.0, 0.1, size=bias.shape))
        for weights, bias in model.layers
    ]
    return model


def ctx_table_rows(model: NeuralEvalModel, board: chess.Board) -> list[int]:
    """Rows of the table that the entries of ``board`` select, as the mover sees them."""
    indices, _ = features_of(model).active(board)
    bucket = next(i - KING_BASE for i in indices if KING_BASE <= i < KING_BASE + KING_BUCKETS)
    king = "king_buckets" in model.families
    piece_rows = MATERIAL_BASE * (KING_BUCKETS if king else 1)
    material_rows = PAWN_FILES_BASE - MATERIAL_BASE if "material" in model.families else 0
    rows = []
    for i in indices:
        if i < MATERIAL_BASE:
            rows.append(i + (bucket * MATERIAL_BASE if king else 0))
        elif i < PAWN_FILES_BASE and "material" in model.families:
            rows.append(piece_rows + i - MATERIAL_BASE)
        elif PAWN_FILES_BASE <= i < KING_BASE and "pawn_files" in model.families:
            rows.append(piece_rows + material_rows + i - PAWN_FILES_BASE)
    return rows


def ctx_logit(model: NeuralEvalModel, board: chess.Board) -> float:
    """Reference of a ``pst_ctx`` network: each view is the position with that side to move."""
    (w0, b0), (w1, b1) = model.layers
    rival = board.copy()
    rival.turn = not board.turn
    views = [ctx_table_rows(model, board), ctx_table_rows(model, rival)]
    h = np.concatenate([np.clip(w0[rows].sum(axis=0) + b0, 0.0, 1.0) for rows in views])
    return float((h @ w1 + b1)[0])


def dense_logit(model: NeuralEvalModel, board: chess.Board) -> float:
    """Reference forward pass on a dense 768- (or 769-) vector, same weights as ``model``."""
    if model.feature_set == "pst_ctx":
        return ctx_logit(model, board)
    indices, values = features_of(model).active(board)
    x = np.zeros(features_of(model).size())
    for index, value in zip(indices, values, strict=True):
        x[index] += value
    (w0, b0), (w1, b1) = model.layers
    h = np.clip(x @ w0 + b0, 0.0, 1.0)
    # the rival's view: every piece in the other block of its square; the dense entries have no
    # rival view
    rival = np.zeros_like(x)
    for index, value in zip(indices, values, strict=True):
        if index < 768:
            rival[(index + 384) % 768] += value
    h = np.concatenate([h, np.clip(rival @ w0 + b0, 0.0, 1.0)])
    return float((h @ w1 + b1)[0])


class TwoHiddenModel(NeuralEvalModel):
    """A 64-16-1 network: a hidden layer after the accumulator, which the core does not have."""

    name = "net_test_x16"
    feature_set = "pst_slotted"
    hidden = (64, 16)


class NarrowModel(NeuralEvalModel):
    """An accumulator of 12 units, not a multiple of 8."""

    name = "net_test_w12"
    feature_set = "pst_slotted"
    hidden = (12,)


@pytest.mark.parametrize("model_class", [TwoHiddenModel, NarrowModel])
def test_a_network_is_one_accumulator_of_a_multiple_of_8_and_an_output(model_class):
    with pytest.raises(ValueError, match="multiple of 8"):
        model_class.layer_shapes()


def test_the_second_layer_reads_both_sides_so_its_input_is_twice_the_accumulator():
    model = new_model(NnW64BothModel, Representation())
    assert [weights.shape for weights, _ in model.layers] == [(768, 64), (128, 1)]
    assert model.count_parameters() == 768 * 64 + 64 + 128 + 1


BOTH_MODELS = [NnW64BothModel, NnThreatsW64BothModel]
CTX_MODELS = [get_model_class(f"net_ctx_{letters}_w128_both") for letters in ("m", "p", "k", "mpk")]


@pytest.mark.parametrize("model_class", [*BOTH_MODELS, *CTX_MODELS])
def test_core_score_matches_a_dense_numpy_reference(model_class):
    model = random_net(seed=6, model_class=model_class)
    evaluator = model.evaluator()
    for board in random_positions(10, seed=8):
        assert evaluator.evaluate(board.fen(), board.turn) == pytest.approx(
            mover_score(dense_logit(model, board))
        )


@pytest.mark.parametrize("model_class", [*BOTH_MODELS, *CTX_MODELS])
def test_the_training_graph_computes_what_the_core_does(model_class):
    from chess_lab.training.epochs import encode_frame
    from chess_lab.training.neural import _assign, _keras_network, _padded, _training_keras

    model = random_net(seed=5, model_class=model_class)
    features = features_of(model)
    width = features.max_entries()
    keras = _training_keras()
    network, parts = _keras_network(
        keras,
        [w.shape for w, _ in model.layers],
        width,
        0.0,
        model.families,
        model.feature_set == "pst_ctx",
    )
    _assign(parts, model.layers)
    boards = random_positions(40, seed=9)
    index, value, _ = _padded(encode_frame(features, Representation(), frame(boards)), width)
    logits = np.asarray(network.predict([index, value], verbose=0)).reshape(-1)
    for board, logit in zip(boards, logits, strict=True):
        assert logit == pytest.approx(dense_logit(model, board), abs=1e-4)


class CtxPlainModel(NeuralEvalModel):
    """The ``pst_ctx`` inputs without any family: only the pieces, as ``pst_slotted``."""

    name = "net_test_ctx_plain"
    feature_set = "pst_ctx"
    hidden = (128,)


def test_ctx_without_families_is_the_slotted_network():
    from chess_lab.training.epochs import encode_frame
    from chess_lab.training.neural import _assign, _keras_network, _padded, _training_keras

    keras = _training_keras()
    boards = random_positions(30, seed=13)
    logits = []
    for model_class, ctx in ((CtxPlainModel, True), (get_model_class("net_pst_w128_both"), False)):
        model = random_net(seed=4, model_class=model_class)
        features = features_of(model)
        width = features.max_entries()
        network, parts = _keras_network(
            keras, [w.shape for w, _ in model.layers], width, 0.0, (), ctx
        )
        _assign(parts, model.layers)
        index, value, _ = _padded(encode_frame(features, Representation(), frame(boards)), width)
        logits.append(np.asarray(network.predict([index, value], verbose=0)).reshape(-1))
    assert logits[0] == pytest.approx(logits[1], abs=1e-5)


@pytest.mark.parametrize("model_class", [NnW64BothModel, CTX_MODELS[-1]])
def test_mirror_complement_holds_for_random_weights(model_class):
    model = random_net(seed=3, model_class=model_class)
    for board in random_positions(20, seed=11):
        assert white_score_of(model, board) == pytest.approx(
            1.0 - white_score_of(model, board.mirror())
        )


def test_fit_stream_rejects_empty_validation():
    model = new_model(NnW64BothModel, Representation())
    with pytest.raises(ValueError, match="validation set is empty"):
        fit_stream(model, [frame(random_positions(10, seed=1))], frame([]), {})


def test_save_load_round_trip_gives_identical_scores(tmp_path):
    rep = Representation(scale="cp_linear")
    model = random_net(seed=4, rep=rep)
    path = model.save(tmp_path / "net_pst_w64_both.npz")

    loaded = NnW64BothModel.load(path)

    assert Representation.from_dict(loaded.rep) == rep
    assert loaded.count_parameters() == model.count_parameters()
    for board in random_positions(10, seed=2):
        assert white_score_of(loaded, board) == pytest.approx(white_score_of(model, board))


def test_a_saved_ctx_network_scores_the_same_after_loading(tmp_path):
    model_class = get_model_class("net_ctx_mpk_w128_both")
    model = random_net(seed=5, model_class=model_class)
    path = model.save(tmp_path / "net_ctx_mpk_w128_both.npz")
    loaded = model_class.load(path)
    for board in random_positions(10, seed=2):
        assert white_score_of(loaded, board) == pytest.approx(white_score_of(model, board))


def test_core_rejects_a_mismatching_input_layer(tmp_path):
    path = tmp_path / "wrong.npz"
    meta = {
        "features": "pst_slotted",
        "rep": Representation().to_dict(),
        "hidden": [64],
    }
    arrays = {"W0": np.zeros((10, 64)), "b0": np.zeros(64)}
    for index, (weights, bias) in enumerate(
        new_model(NnW64BothModel, Representation()).layers[1:], start=1
    ):
        arrays[f"W{index}"], arrays[f"b{index}"] = weights, bias
    with open(path, "wb") as handle:
        np.savez(handle, meta=np.array(json.dumps(meta)), **arrays)
    with pytest.raises(ValueError, match="768"):
        NnW64BothModel.load(path).evaluator()


def test_play_path_does_not_import_tensorflow(tmp_path):
    path = random_net(seed=7, model_class=NnW64BothModel).save(tmp_path / "net_pst_w64_both.npz")
    script = (
        "import sys, chess\n"
        "from chess_lab.models.definitions import load_evaluator\n"
        f"evaluator = load_evaluator('net_pst_w64_both', {str(path)!r})\n"
        "score = evaluator.evaluate(chess.Board().fen(), True)\n"
        "print(0.0 < score < 1.0, 'tensorflow' in sys.modules, 'keras' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.split() == ["True", "False", "False"]
