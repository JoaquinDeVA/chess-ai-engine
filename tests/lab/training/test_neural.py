"""Tests of the sparse-input neural evaluation core."""

import json
import subprocess
import sys

import numpy as np
import pytest

from chess_lab.models.definitions import (
    NeuralEvalModel,
    NnW64BothModel,
    NnW128Ob8BothModel,
    NnW128Ob8H16BothModel,
    get_model_class,
)
from chess_lab.models.features import (
    features_of,
)
from chess_lab.models.target import Representation
from chess_lab.training.neural import checkpoint_config, fit_stream, new_model, warm_layers
from tests.conftest import CTX_MODELS, random_net, random_positions, white_score_of
from tests.lab.models.positions import frame


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


def test_a_head_gives_every_bucket_its_own_hidden_layer():
    model = new_model(NnW128Ob8H16BothModel, Representation())
    assert [weights.shape for weights, _ in model.layers] == [(768, 128), (256, 128), (16, 8)]
    assert model.count_parameters() == 768 * 128 + 128 + 256 * 128 + 128 + 16 * 8 + 8


def test_eight_output_buckets_add_eight_output_units():
    model = new_model(NnW128Ob8BothModel, Representation())
    assert [weights.shape for weights, _ in model.layers] == [(768, 128), (256, 8)]


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


WARM_TARGETS = [get_model_class("net_pst_w128_both"), NnW128Ob8BothModel, *CTX_MODELS]


@pytest.mark.parametrize("model_class", WARM_TARGETS, ids=lambda cls: cls.name)
def test_a_warm_start_scores_as_its_source_in_the_core(model_class):
    source = random_net(seed=6, model_class=get_model_class("net_pst_w128_both"))
    (weights, bias), (output_weights, output_bias) = source.layers
    source.layers = [(weights, bias), (0.1 * output_weights, output_bias)]  # off the score clip
    warm = model_class(layers=warm_layers(source, model_class), rep=source.rep)
    assert [w.shape for w, _ in warm.layers] == [shape for shape in model_class.layer_shapes()]
    boards = random_positions(40, seed=17)
    scores = [white_score_of(source, board) for board in boards]
    assert 0.05 < min(scores) and max(scores) < 0.95
    assert [white_score_of(warm, board) for board in boards] == pytest.approx(scores, abs=1e-6)


@pytest.mark.parametrize(
    "name", ["net_pst_w256_both", "net_pst_threats_w64_both", "net_pst_w128_ob8h16_both"]
)
def test_a_warm_start_refuses_another_width_feature_set_or_a_head(name):
    source = random_net(model_class=get_model_class("net_pst_w128_both"))
    with pytest.raises(ValueError, match="cannot start"):
        warm_layers(source, get_model_class(name))


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


class CutAfterOneEpoch:
    """Training frames whose second reading fails, as a run killed during its second epoch."""

    def __init__(self, frames):
        self.frames, self.readings = frames, 0

    def __iter__(self):
        self.readings += 1
        if self.readings > 1:
            raise RuntimeError("cut")
        return iter(self.frames)


@pytest.mark.parametrize("model_class", [NnW64BothModel, NnW128Ob8H16BothModel])
def test_a_fit_cut_after_an_epoch_resumes_as_if_it_had_not_stopped(tmp_path, model_class):
    boards = random_positions(64, seed=21)
    training, validation = [frame(boards[:48])], frame(boards[48:])
    config = {"max_epochs": 2, "lr_decay": 0.5, "patience": 5, "batch_size": 16}
    whole = new_model(model_class, Representation())
    expected = fit_stream(whole, training, validation, config)

    checkpoint = tmp_path / "fit.npz"
    with pytest.raises(RuntimeError, match="cut"):
        fit_stream(
            new_model(model_class, Representation()),
            CutAfterOneEpoch(training),
            validation,
            config,
            checkpoint=checkpoint,
        )
    assert checkpoint_config(checkpoint)["max_epochs"] == 2
    resumed = new_model(model_class, Representation())
    summary = fit_stream(resumed, training, validation, config, checkpoint=checkpoint)

    for key in ("epochs", "best_epoch", "train_rows"):
        assert summary[key] == expected[key]
    for key in ("train_losses", "val_losses"):
        assert np.allclose(summary[key], expected[key])
    for (weights, bias), (want_weights, want_bias) in zip(
        resumed.layers, whole.layers, strict=True
    ):
        assert np.allclose(weights, want_weights, atol=1e-6)
        assert np.allclose(bias, want_bias, atol=1e-6)


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
