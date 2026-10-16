"""Tests of the linear evaluation core and of the tapered piece-square-table model."""

import json
import os

import chess
import numpy as np
import pandas as pd
import pytest

from chess_engine.models.linear_eval import LinearEvalModel
from chess_engine.models.linear_features import MaterialCountFeatures, PstFeatures, phase
from chess_engine.models.registry import available_models, get_model_class
from chess_engine.training.representation import Representation
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


def random_model(seed: int = 0, rep: Representation | None = None) -> LinearEvalModel:
    """Pst model with random weights, so that scores are not trivially 0.5."""
    model = get_model_class("pst_tapered")(rep=rep)
    rng = np.random.default_rng(seed)
    model.weights = rng.normal(0.0, 0.3, size=model.weights.shape)
    model.bias = 0.1
    return model


def test_phase_is_one_at_start_and_zero_with_bare_kings():
    assert phase(chess.Board()) == pytest.approx(1.0)
    assert phase(chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")) == pytest.approx(0.0)


def test_pst_features_are_antisymmetric():
    for board in random_positions(20, seed=7):
        indices, values = PstFeatures().active(board, "side_to_move")
        m_indices, m_values = PstFeatures().active(board.mirror(), "side_to_move")
        assert sorted(zip(indices, values, strict=True)) == sorted(
            zip(m_indices, m_values, strict=True)
        )


def test_mirrored_position_gets_complementary_white_score():
    model = random_model(seed=3)
    for board in random_positions(20, seed=11):
        assert model.expected_score(board) == pytest.approx(
            1.0 - model.expected_score(board.mirror())
        )


def test_fit_learns_material():
    model = get_model_class("pst_tapered")()
    boards = random_positions(3000, seed=5)
    batches = [frame(boards[i : i + 500]) for i in range(0, len(boards), 500)]
    config = {"l2": 0.0, "lr": 0.05, "batch_size": 256, "max_epochs": 60, "patience": 60}
    summary = model.fit_stream(batches, frame(random_positions(300, seed=9)), config)

    assert summary["epochs"] >= 1
    assert model.expected_score(chess.Board(WHITE_UP_A_QUEEN)) > 0.9
    assert model.expected_score(chess.Board(BLACK_UP_A_QUEEN)) < 0.1


def test_save_load_round_trip_keeps_scores_and_representation(tmp_path):
    rep = Representation(scale="cp_linear", perspective="white", filter="quiet")
    model = random_model(seed=4, rep=rep)
    path = model.save(tmp_path / "model.npz")

    loaded = get_model_class("pst_tapered").load(path)

    assert loaded.rep == rep
    assert loaded.count_parameters() == model.count_parameters() == 769 + 1
    for board in random_positions(10, seed=2):
        assert loaded.expected_score(board) == pytest.approx(model.expected_score(board))
        assert loaded.predict_cp(board) == pytest.approx(model.predict_cp(board))


def test_logit_matches_dense_computation():
    model = random_model(seed=6)
    for board in random_positions(10, seed=8):
        indices, values = model.features.active(board, model.rep.perspective)
        dense = np.zeros(model.weights.shape)
        for index, value in zip(indices, values, strict=True):
            dense[index] += value
        assert model.logit(board) == pytest.approx(float(model.weights @ dense + model.bias))


def test_count_parameters_matches_size_plus_bias():
    assert get_model_class("pst_tapered")().count_parameters() == 768 + 1


def test_fit_rejects_empty_validation():
    model = get_model_class("pst_tapered")()
    with pytest.raises(ValueError, match="validation set is empty"):
        model.fit_stream([frame(random_positions(10, seed=1))], frame([]), {})


def test_fit_rejects_non_finite_validation_loss():
    model = get_model_class("pst_tapered")()
    boards = random_positions(40, seed=12)
    config = {"lr": float("nan"), "max_epochs": 2}
    with pytest.raises(ValueError, match="not finite"):
        model.fit_stream([frame(boards)], frame(random_positions(10, seed=13)), config)


def test_workers_give_the_same_weights_as_one_worker():
    boards = random_positions(120, seed=14)
    batches = [frame(boards[i : i + 40]) for i in range(0, len(boards), 40)]
    validation = frame(random_positions(30, seed=15))
    config = {"batch_size": 16, "max_epochs": 3, "patience": 3, "lr": 0.05}
    env_before = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}

    serial = get_model_class("pst_tapered")()
    parallel = get_model_class("pst_tapered")()
    serial_summary = serial.fit_stream(batches, validation, {**config, "workers": 1})
    parallel_summary = parallel.fit_stream(batches, validation, {**config, "workers": 2})

    assert np.array_equal(serial.weights, parallel.weights)
    assert serial.bias == parallel.bias
    assert serial_summary["val_losses"] == parallel_summary["val_losses"]
    assert serial_summary["train_losses"] == parallel_summary["train_losses"]
    env_after = {name: os.environ.get(name) for name in env_before}
    assert env_after == env_before


def test_base_class_refuses_save_and_load(tmp_path):
    with pytest.raises(TypeError, match="defines no features"):
        LinearEvalModel()
    with pytest.raises(TypeError, match="defines no features"):
        LinearEvalModel.load(tmp_path / "missing.npz")


def test_load_rejects_weights_of_the_wrong_size(tmp_path):
    path = tmp_path / "wrong.npz"
    meta = {"features": "pst_tapered", "rep": Representation().to_dict()}
    with open(path, "wb") as handle:
        np.savez(
            handle,
            weights=np.zeros(10),
            bias=np.float64(0.0),
            meta=np.array(json.dumps(meta)),
        )
    with pytest.raises(ValueError, match="expected 768 weights.*found 10"):
        get_model_class("pst_tapered").load(path)


MATERIAL_BOARD = "4k3/8/8/8/8/8/PPB5/R3K3 w - - 0 1"  # white: 2 pawns, 1 bishop, 1 rook


def test_material_features_count_differences():
    board = chess.Board(MATERIAL_BOARD)
    # Order: P, N, B, R, Q. Under "white" a turn entry (index 5) follows: +1, White to move.
    indices, values = MaterialCountFeatures().active(board, "white")
    assert indices == [0, 1, 2, 3, 4, 5]
    assert values == [2.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    # Under "side_to_move" the counts are seen from the mover; no turn entry is needed.
    indices, values = MaterialCountFeatures().active(board, "side_to_move")
    assert indices == [0, 1, 2, 3, 4]
    assert values == [2.0, 0.0, 1.0, 1.0, 0.0]
    black_to_move = chess.Board(MATERIAL_BOARD)
    black_to_move.turn = chess.BLACK
    indices, values = MaterialCountFeatures().active(black_to_move, "side_to_move")
    assert values == [-2.0, 0.0, -1.0, -1.0, 0.0]


def test_material_model_mirror_complement():
    model = get_model_class("material_linear")(weights=np.array([0.3, 0.1, 0.2, 0.5, 0.9]))
    model.bias = 0.05
    for board in random_positions(20, seed=11):
        assert model.expected_score(board) == pytest.approx(
            1.0 - model.expected_score(board.mirror())
        )


def test_linear_baseline_is_gone():
    assert "linear_baseline" not in available_models()
