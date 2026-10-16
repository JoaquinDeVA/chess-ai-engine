"""Tests of the model layer: feature extractors, registry, scaling and the shipped artifacts."""

import chess
import numpy as np
import pytest

from chess_engine.models import (
    CP_SCALE,
    available_models,
    cp_to_expected_score,
    get_model_class,
    load_model,
)
from chess_engine.models.base import ARTIFACTS_DIR, EvaluationModel
from chess_engine.models.evaluator import ModelEvaluator
from chess_engine.models.features import BitboardFeatures, MaterialFeatures, VectorFeatures
from chess_engine.models.registry import register_model
from chess_engine.search.chess_game_state import ChessGameState
from tests.conftest import material_cp, random_positions

WHITE_UP_A_QUEEN = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_UP_A_QUEEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"


def reference_vector(board: chess.Board) -> np.ndarray:
    """Verbatim copy of the feature code the course models were trained with."""
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    center = {chess.D4, chess.D5, chess.E4, chess.E5}
    material = [0.0, 0.0]
    piece_count = np.zeros((2, 5), dtype=np.float32)
    pawn_rank_sum = [0.0, 0.0]
    pawn_count = [0, 0]
    center_control = [0, 0]
    king_square = [None, None]
    for square, piece in board.piece_map().items():
        color = int(piece.color)
        ptype = piece.piece_type
        if ptype != chess.KING:
            material[color] += values[ptype]
            piece_count[color, ptype - 1] += 1
        if ptype == chess.PAWN:
            pawn_count[color] += 1
            pawn_rank_sum[color] += chess.square_rank(square) / 7.0
        if square in center:
            center_control[color] += 1
        if ptype == chess.KING:
            king_square[color] = square
    material_diff = material[0] - material[1]
    phase = (material[0] + material[1]) / 78.0
    pawn_rank_avg = [pawn_rank_sum[c] / pawn_count[c] if pawn_count[c] > 0 else 0.0 for c in (0, 1)]

    def king_activity(square):
        if square is None:
            return 0.0
        return (abs(chess.square_file(square) - 3.5) + abs(chess.square_rank(square) - 3.5)) / 7.0

    return np.array(
        [
            1.0 if board.turn == chess.WHITE else -1.0,
            phase,
            1.0 - phase,
            material[0] / 39.0,
            material[1] / 39.0,
            material_diff / 39.0,
            *(piece_count[0] / np.array([8, 2, 2, 2, 1])),
            *(piece_count[1] / np.array([8, 2, 2, 2, 1])),
            pawn_rank_avg[0],
            pawn_rank_avg[1],
            pawn_count[0] / 8.0,
            pawn_count[1] / 8.0,
            center_control[0] / 4.0,
            center_control[1] / 4.0,
            king_activity(king_square[0]),
            king_activity(king_square[1]),
            board.is_check() and board.turn == chess.WHITE,
            board.is_check() and board.turn == chess.BLACK,
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.legal_moves.count() / 50.0,
        ],
        dtype=np.float32,
    )


class TestFeatures:
    def test_vector_features_match_the_layout_the_models_were_trained_with(self, sample_boards):
        extractor = VectorFeatures()
        for board in sample_boards:
            np.testing.assert_allclose(extractor.encode(board)[0], reference_vector(board))

    def test_vector_has_29_float32_numbers(self):
        (vector,) = VectorFeatures().encode(chess.Board())
        assert vector.shape == (29,) and vector.dtype == np.float32

    def test_bitboard_planes_and_extras(self):
        tensor, extras = BitboardFeatures().encode(chess.Board())
        assert tensor.shape == (8, 8, 12) and extras.shape == (9,)
        assert tensor[6, :, 0].sum() == 8, "white pawns on rank 2 (row 6), plane 0"
        assert tensor[1, :, 6].sum() == 8, "black pawns on rank 7 (row 1), plane 6"
        assert tensor.sum() == 32
        assert extras[0] == 1.0 and extras[5] == extras[6] == 1.0

    def test_material_features_are_white_minus_black(self):
        (vector,) = MaterialFeatures().encode(chess.Board(WHITE_UP_A_QUEEN))
        assert vector.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def test_encode_batch_stacks_every_input(self, sample_boards):
        tensor, extras = BitboardFeatures().encode_batch(sample_boards[:5])
        assert tensor.shape == (5, 8, 8, 12) and extras.shape == (5, 9)

    def test_encode_batch_of_nothing_keeps_the_input_shapes(self):
        (vectors,) = VectorFeatures().encode_batch([])
        assert vectors.shape == (0, 29)

    def test_encode_fens_accepts_four_field_fens(self):
        (vectors,) = VectorFeatures().encode_fens(
            ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"]
        )
        assert vectors.shape == (1, 29)


class TestScale:
    def test_expected_score_is_centred_bounded_and_monotonic(self):
        assert cp_to_expected_score(0) == 0.5
        assert cp_to_expected_score(CP_SCALE) == cp_to_expected_score(10 * CP_SCALE) == 0.99
        assert cp_to_expected_score(-CP_SCALE) == 0.01
        scores = [cp_to_expected_score(cp) for cp in range(-1500, 1501, 100)]
        assert scores == sorted(scores)

    def test_expected_score_is_antisymmetric_around_one_half(self):
        for cp in (10, 250, 800):
            assert cp_to_expected_score(cp) + cp_to_expected_score(-cp) == pytest.approx(1.0)

    def test_evaluations_never_reach_the_terminal_values(self):
        """Regression B10: the search compares evaluations with terminal values 0, 0.5 and 1."""
        assert 0 < cp_to_expected_score(-1e9) and cp_to_expected_score(1e9) < 1


class TestRegistry:
    def test_builtin_models_are_registered(self):
        assert {
            "ridge_features",
            "mlp_features",
            "cnn_resnet",
            "linear_baseline",
        } <= set(available_models())

    def test_unknown_model_lists_the_valid_names(self):
        with pytest.raises(KeyError, match="available:.*linear_baseline"):
            get_model_class("nope")

    def test_registering_a_name_twice_is_an_error(self):
        with pytest.raises(ValueError, match="already registered"):

            @register_model("linear_baseline")
            class Impostor(EvaluationModel):  # pragma: no cover - never instantiated
                pass

    def test_missing_artifact_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model("linear_baseline", tmp_path / "missing.pkl")

    def test_frozen_models_cannot_be_retrained(self):
        assert not get_model_class("cnn_light").trainable
        assert get_model_class("ridge_features").trainable


class TestShippedModels:
    @pytest.mark.parametrize("name", ["mlp_features", "ridge_features", "cnn_light", "cnn_resnet"])
    def test_sign_is_white_point_of_view(self, name):
        pytest.importorskip("keras")
        model = load_model(name)
        assert model.predict_cp(chess.Board(WHITE_UP_A_QUEEN)) > model.predict_cp(
            chess.Board(BLACK_UP_A_QUEEN)
        )

    def test_parameter_counts_match_the_architectures(self):
        pytest.importorskip("keras")
        assert load_model("mlp_features").count_parameters() == 1505
        assert load_model("cnn_light").count_parameters() == 109_569
        assert load_model("cnn_resnet").count_parameters() == 572_353

    def test_numpy_inference_of_the_mlp_matches_keras(self, sample_boards):
        pytest.importorskip("keras")
        model = load_model("mlp_features")
        inputs = model.extractor.encode_batch(sample_boards)
        expected = np.asarray(model.network(inputs[0], training=False)).ravel()
        np.testing.assert_allclose(model.predict_target(inputs), expected, atol=1e-4)

    def test_cnn_resnet_architecture_reproduces_the_delivered_network(self):
        pytest.importorskip("keras")
        assert get_model_class("cnn_resnet").new().count_parameters() == 572_353

    def test_batch_prediction_matches_single_predictions(self, sample_boards):
        pytest.importorskip("keras")
        model = load_model("cnn_light")
        batch = model.predict_cp_batch(sample_boards[:8])
        single = [model.predict_cp(board) for board in sample_boards[:8]]
        np.testing.assert_allclose(batch, single, atol=1e-2)


class TestLinearBaseline:
    def test_fit_save_load_roundtrip_learns_material(self, tmp_path, sample_boards):
        from tests.conftest import random_positions

        boards = random_positions(400, seed=5)
        model = get_model_class("linear_baseline").new()
        inputs = model.extractor.encode_batch(boards)
        targets = np.clip(np.array([material_cp(b) for b in boards]) / CP_SCALE, -1, 1)
        summary = model.fit(inputs, targets)
        assert summary["train_samples"] == 400

        path = model.save(tmp_path / "baseline.pkl")
        loaded = get_model_class("linear_baseline").load(path)
        predictions = loaded.predict_cp_batch(sample_boards)
        truth = np.clip([material_cp(b) for b in sample_boards], -CP_SCALE, CP_SCALE)
        assert np.corrcoef(predictions, truth)[0, 1] > 0.9
        assert loaded.count_parameters() == 7  # 6 coefficients + intercept

    def test_predictions_are_clipped_to_the_cp_scale(self):
        model = get_model_class("linear_baseline").new()
        boards = [chess.Board(WHITE_UP_A_QUEEN), chess.Board(BLACK_UP_A_QUEEN)]
        model.fit(model.extractor.encode_batch(boards), np.array([1.0, -1.0]))
        extreme = chess.Board("4k3/8/8/8/8/8/PPPPPPPP/QQQQKQQQ w - - 0 1")
        assert abs(model.predict_cp(extreme)) <= CP_SCALE


@pytest.mark.skipif(
    not (ARTIFACTS_DIR / "mlp_features.keras").exists(), reason="mlp_features not trained"
)
def test_trained_mlp_agrees_with_material_on_clear_cases():
    pytest.importorskip("keras")
    model = load_model("mlp_features")
    assert model.predict_cp(chess.Board(WHITE_UP_A_QUEEN)) > model.predict_cp(
        chess.Board(BLACK_UP_A_QUEEN)
    )


def test_evaluator_unchanged_for_existing_models(stub_model):
    evaluator = ModelEvaluator(stub_model)
    for board in random_positions(30, seed=21):
        cp = stub_model.predict_cp(board)
        for player in (chess.WHITE, chess.BLACK):
            expected = cp_to_expected_score(cp if player == chess.WHITE else -cp)
            value = evaluator(ChessGameState(board), player)
            assert value == pytest.approx(expected, abs=1e-12)
