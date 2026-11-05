"""Tests of the linear evaluation core and of the tapered piece-square-table model."""

import json
import os

import chess
import numpy as np
import pytest

from chess_lab.models import fit_stream, linear_eval, new_model
from chess_lab.models.definitions import (
    LinearEvalModel,
    available_models,
    get_model_class,
    load_evaluator,
)
from chess_lab.models.linear_features import (
    _PAWN_CACHE,
    BLOCK_SIZE,
    FEATURE_SETS,
    TERM_NAMES,
    KingSlottedPstFeatures,
    PstFeatures,
    PstTermsFeatures,
    SlottedPstFeatures,
    _pawn_terms,
    features_of,
    phase,
)
from chess_lab.models.representation import Representation
from tests.conftest import mover_score, random_positions, white_score_of
from tests.lab.models.positions import BLACK_UP_A_QUEEN, WHITE_UP_A_QUEEN, frame


def random_model(seed: int = 0, rep: Representation | None = None) -> LinearEvalModel:
    """Pst model with random weights, so that scores are not trivially 0.5."""
    model = new_model(get_model_class("linear_pst_terms"), rep)
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
        assert white_score_of(model, board) == pytest.approx(
            1.0 - white_score_of(model, board.mirror())
        )


@pytest.mark.slow
def test_fit_learns_material():
    model = new_model(get_model_class("linear_pst_terms"), Representation())
    boards = random_positions(3000, seed=5)
    batches = [frame(boards[i : i + 500]) for i in range(0, len(boards), 500)]
    config = {"l2": 0.0, "lr": 0.05, "batch_size": 256, "max_epochs": 60, "patience": 60}
    summary = fit_stream(model, batches, frame(random_positions(300, seed=9)), config)

    assert summary["epochs"] >= 1
    assert white_score_of(model, chess.Board(WHITE_UP_A_QUEEN)) > 0.9
    assert white_score_of(model, chess.Board(BLACK_UP_A_QUEEN)) < 0.1


def test_save_load_round_trip_keeps_scores_and_representation(tmp_path):
    rep = Representation(scale="cp_linear", perspective="white")
    model = random_model(seed=4, rep=rep)
    path = model.save(tmp_path / "model.npz")

    loaded = get_model_class("linear_pst_terms").load(path)

    assert Representation.from_dict(loaded.rep) == rep
    assert loaded.count_parameters() == model.count_parameters() == 791 + 1
    for board in random_positions(10, seed=2):
        assert white_score_of(loaded, board) == pytest.approx(white_score_of(model, board))


def test_core_score_matches_dense_computation():
    model = random_model(seed=6)
    for board in random_positions(10, seed=8):
        indices, values = features_of(model).active(board, model.perspective)
        dense = np.zeros(model.weights.shape)
        for index, value in zip(indices, values, strict=True):
            dense[index] += value
        assert model.evaluator().evaluate(board.fen(), board.turn) == pytest.approx(
            mover_score(float(model.weights @ dense + model.bias))
        )


def test_count_parameters_matches_size_plus_bias():
    assert (
        new_model(get_model_class("linear_pst_terms"), Representation()).count_parameters()
        == 790 + 1
    )


def test_fit_rejects_empty_validation():
    model = new_model(get_model_class("linear_pst_terms"), Representation())
    with pytest.raises(ValueError, match="validation set is empty"):
        fit_stream(model, [frame(random_positions(10, seed=1))], frame([]), {})


def test_fit_rejects_non_finite_validation_loss():
    model = new_model(get_model_class("linear_pst_terms"), Representation())
    boards = random_positions(40, seed=12)
    config = {"lr": float("nan"), "max_epochs": 2}
    with pytest.raises(ValueError, match="not finite"):
        fit_stream(model, [frame(boards)], frame(random_positions(10, seed=13)), config)


def test_workers_give_the_same_weights_as_one_worker():
    boards = random_positions(120, seed=14)
    batches = [frame(boards[i : i + 40]) for i in range(0, len(boards), 40)]
    validation = frame(random_positions(30, seed=15))
    config = {"batch_size": 16, "max_epochs": 3, "patience": 3, "lr": 0.05}
    env_before = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}

    serial = new_model(get_model_class("linear_pst_terms"), Representation())
    parallel = new_model(get_model_class("linear_pst_terms"), Representation())
    serial_summary = fit_stream(serial, batches, validation, {**config, "workers": 1})
    parallel_summary = fit_stream(parallel, batches, validation, {**config, "workers": 2})

    assert np.array_equal(serial.weights, parallel.weights)
    assert serial.bias == parallel.bias
    assert serial_summary["val_losses"] == parallel_summary["val_losses"]
    assert serial_summary["train_losses"] == parallel_summary["train_losses"]
    env_after = {name: os.environ.get(name) for name in env_before}
    assert env_after == env_before


def test_core_rejects_weights_of_the_wrong_size(tmp_path):
    path = tmp_path / "wrong.npz"
    meta = {"features": "linear_pst_terms", "rep": Representation().to_dict()}
    with open(path, "wb") as handle:
        np.savez(
            handle,
            weights=np.zeros(10),
            bias=np.float64(0.0),
            meta=np.array(json.dumps(meta)),
        )
    with pytest.raises(ValueError, match="790"):
        load_evaluator("linear_pst_terms", path)


def test_linear_baseline_is_gone():
    assert "linear_baseline" not in available_models()


def term_values(board: chess.Board) -> dict[str, float]:
    """White-POV value of every classical term (middlegame plus endgame weight of its entries)."""
    indices, values = PstTermsFeatures().active(board, "white")
    totals = dict.fromkeys(TERM_NAMES, 0.0)
    for index, value in zip(indices, values, strict=True):
        if 2 * BLOCK_SIZE <= index < 2 * BLOCK_SIZE + 2 * len(TERM_NAMES):
            totals[TERM_NAMES[(index - 2 * BLOCK_SIZE) // 2]] += value
    return totals


def test_passed_pawn_term():
    # Lone white pawn on a2 with no black pawn on files a and b: it is passed.
    assert term_values(chess.Board("4k3/8/8/8/8/8/P7/4K3 w - - 0 1"))["passed"] == pytest.approx(
        0.25
    )
    # A black pawn on b5 is ahead on an adjacent file: nobody is passed.
    blocked = term_values(chess.Board("4k3/8/8/1p6/8/8/P7/4K3 w - - 0 1"))
    assert blocked["passed"] == pytest.approx(0.0)


def test_doubled_and_isolated():
    # a2 and a3 are doubled; neither has a friendly pawn on file b, so both are isolated.
    totals = term_values(chess.Board("4k3/8/8/8/8/P7/P7/4K3 w - - 0 1"))
    assert totals["doubled"] == pytest.approx(0.25)
    assert totals["isolated"] == pytest.approx(0.5)


def test_bishop_pair():
    assert term_values(chess.Board("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1"))["bishop_pair"] == 1.0
    assert term_values(chess.Board("4k3/8/8/8/8/8/8/2B1K3 w - - 0 1"))["bishop_pair"] == 0.0
    # Black's pair counts against White.
    assert term_values(chess.Board("2b1kb2/8/8/8/8/8/8/4K3 w - - 0 1"))["bishop_pair"] == -1.0


def test_rook_open_and_half_open_file():
    # Rook a1 sits on an empty file; rook e1 faces only the black pawn on e7.
    totals = term_values(chess.Board("4k3/4p3/8/8/8/8/8/R3R1K1 w - - 0 1"))
    assert totals["rook_open"] == 1.0
    assert totals["rook_half_open"] == 1.0


def test_king_shield():
    # Three own pawns on f2, g2, h2 in front of the king on g1: 3 / 3.
    shield = term_values(chess.Board("4k3/8/8/8/8/8/5PPP/6K1 w - - 0 1"))["king_shield"]
    assert shield == pytest.approx(1.0)
    # A pawn on g4 is more than two ranks ahead of the king: it does not count.
    far = term_values(chess.Board("4k3/8/8/8/6P1/8/8/6K1 w - - 0 1"))["king_shield"]
    assert far == pytest.approx(0.0)


def test_mobility_counts_attacks_not_legal_moves():
    # The knight on e2 is pinned by the rook on e8, so it has no legal move, but it attacks
    # six squares, none of them occupied by White: 6 / 10.
    board = chess.Board("4r2k/8/8/8/8/8/4N3/4K3 w - - 0 1")
    assert not any(move.from_square == chess.E2 for move in board.legal_moves)
    assert term_values(board)["mobility_knight"] == pytest.approx(0.6)


def test_pst_terms_mirror_complement():
    model = new_model(get_model_class("linear_pst_terms"), Representation())
    rng = np.random.default_rng(3)
    model.weights = rng.normal(0.0, 0.3, size=model.weights.shape)
    model.bias = 0.1
    for board in random_positions(20, seed=17):
        assert white_score_of(model, board) == pytest.approx(
            1.0 - white_score_of(model, board.mirror())
        )


def test_pst_terms_parameter_count():
    # 768 PST weights, two per term (middlegame and endgame), plus the bias.
    assert (
        new_model(get_model_class("linear_pst_terms"), Representation()).count_parameters()
        == 768 + 2 * 11 + 1
    )


def test_pawn_cache_hit_returns_same_features():
    _PAWN_CACHE.clear()
    first = chess.Board("4k3/8/8/8/8/8/P7/4K3 w - - 0 1")
    # Same pawns, king elsewhere: the pawn terms come from the cache.
    second = chess.Board("4k3/8/8/8/8/8/P7/3K4 w - - 0 1")
    _pawn_terms(first)
    cached = _pawn_terms(first)
    assert _pawn_terms(second) is cached
    assert len(_PAWN_CACHE) == 1


def test_slotted_features_are_one_hot_and_mover_relative():
    features = SlottedPstFeatures()
    indices, values = features.active(chess.Board(), "side_to_move")
    assert len(indices) == 32
    assert values == [1.0] * 32
    assert features.size("side_to_move") == 768
    assert features.max_active == 32
    assert sorted(indices) == sorted(features.active(chess.Board().mirror(), "side_to_move")[0])
    for board in random_positions(200, seed=21):
        indices, _ = features.active(board, "side_to_move")
        m_indices, _ = features.active(board.mirror(), "side_to_move")
        assert sorted(indices) == sorted(m_indices)
        assert len(indices) <= features.max_entries("side_to_move")
        white_indices, _ = features.active(board, "white")
        assert len(white_indices) <= features.max_entries("white")


def test_slotted_white_perspective_has_turn_entry():
    features = SlottedPstFeatures()
    assert features.size("white") == 769
    assert features.max_entries("white") == 33
    assert features.max_entries("side_to_move") == 32
    indices, values = features.active(chess.Board(), "white")
    assert len(indices) == 33
    assert sum(1 for index in indices if index < 384) == 16  # White pieces
    assert sum(1 for index in indices if 384 <= index < 768) == 16  # Black pieces
    assert indices[-1] == 768
    assert values[-1] == 1.0
    black_to_move = chess.Board()
    black_to_move.turn = chess.BLACK
    assert SlottedPstFeatures().active(black_to_move, "white")[1][-1] == 0.0


def _cache_fit(cache_dir):
    boards = random_positions(120, seed=21)
    batches = [frame(boards[i : i + 40]) for i in range(0, len(boards), 40)]
    validation = frame(random_positions(30, seed=22))
    config = {"batch_size": 16, "max_epochs": 3, "patience": 3, "lr": 0.05}
    model = new_model(get_model_class("linear_pst_terms"), Representation())
    fit_stream(model, batches, validation, config, cache_dir=cache_dir)
    return model


def test_cache_gives_the_same_weights_as_no_cache(tmp_path):
    plain = _cache_fit(None)
    cached = _cache_fit(tmp_path / "cache")

    np.testing.assert_allclose(cached.weights, plain.weights, atol=0)
    np.testing.assert_allclose(cached.bias, plain.bias, atol=0)
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 3
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_second_run_with_cache_does_not_encode(tmp_path, monkeypatch):
    first = _cache_fit(tmp_path)

    real = linear_eval.encode_frame

    def only_validation(features, rep, frame_):
        # The 30-row validation set is always encoded; a 40-row training batch never may be.
        if len(frame_) != 30:
            raise AssertionError("encoded a training batch despite the cache")
        return real(features, rep, frame_)

    monkeypatch.setattr(linear_eval, "encode_frame", only_validation)
    second = _cache_fit(tmp_path)

    np.testing.assert_allclose(second.weights, first.weights, atol=0)
    np.testing.assert_allclose(second.bias, first.bias, atol=0)


def _flipped(boards):
    """Validation that contradicts the training data: training makes its loss worse."""
    rows = frame(boards)
    rows["cp"] = -rows["cp"]
    return rows


def _decay_fit(config):
    boards = random_positions(120, seed=31)
    batches = [frame(boards[i : i + 40]) for i in range(0, len(boards), 40)]
    validation = _flipped(random_positions(60, seed=32))
    model = new_model(get_model_class("linear_pst_terms"), Representation())
    summary = fit_stream(model, batches, validation, {"batch_size": 16, "lr": 0.1, **config})
    return model, validation, summary


def test_lr_decay_keeps_training_with_a_smaller_rate_until_max_epochs():
    _, _, summary = _decay_fit({"max_epochs": 6, "patience": 0, "lr_decay": 0.5, "min_lr": 1e-9})

    assert summary["epochs"] == 6  # patience 0 does not cut the run
    history, losses = summary["lr_history"], summary["val_losses"]
    assert len(history) == 6 and history[0] == 0.1
    assert losses[1] >= losses[0]  # the second epoch does not improve: the rate drops from there
    assert history[1] == 0.1 and history[2] == pytest.approx(0.05)
    # Rule: an epoch without improvement multiplies the rate by 0.5; one that improves keeps it.
    best = losses[0]
    for epoch in range(1, 5):
        if losses[epoch] < best:
            best = losses[epoch]
            assert history[epoch + 1] == history[epoch]
        else:
            assert history[epoch + 1] == pytest.approx(history[epoch] * 0.5)


def test_decay_patience_stops_after_that_many_epochs_in_a_row_without_improvement():
    config = {
        "max_epochs": 20,
        "patience": 1,
        "decay_patience": True,
        "lr_decay": 0.5,
        "min_lr": 1e-9,
    }
    _, _, summary = _decay_fit(config)

    assert summary["epochs"] == 2  # epoch 2 does not improve: one in a row


def test_lr_decay_stops_when_the_rate_falls_below_min_lr():
    _, _, summary = _decay_fit({"max_epochs": 20, "patience": 0, "lr_decay": 0.5, "min_lr": 0.06})

    # Epoch 2 does not improve: 0.1 * 0.5 = 0.05 < 0.06, and the run ends there.
    assert summary["epochs"] == 2
    assert summary["lr_history"] == [0.1, 0.1]


def test_without_lr_decay_patience_still_stops_and_the_rate_is_constant():
    _, _, summary = _decay_fit({"max_epochs": 10, "patience": 1})

    assert summary["epochs"] == 3  # best at 1, no improvement at 2 and 3 (> patience)
    assert summary["lr_history"] == [0.1, 0.1, 0.1]


def test_lr_decay_restores_the_weights_of_the_best_epoch():
    model, validation, summary = _decay_fit(
        {"max_epochs": 6, "patience": 0, "lr_decay": 0.5, "min_lr": 1e-9}
    )

    best = summary["best_epoch"]
    assert summary["val_losses"][best - 1] == min(summary["val_losses"])
    rows, cols, vals, targets = linear_eval.encode_frame(
        features_of(model), Representation.from_dict(model.rep), validation
    )
    theta = np.append(model.weights, model.bias)
    z = linear_eval.scores(rows, cols, vals, len(targets), theta)
    restored = float(np.mean(linear_eval.losses(z, targets)))
    assert restored == pytest.approx(summary["val_losses"][best - 1])


def test_king_slotted_features_move_the_slotted_indices_into_the_own_king_block():
    features = KingSlottedPstFeatures()
    plain = SlottedPstFeatures()
    assert features.size("side_to_move") == 64 * 768
    assert features.size("white") == 64 * 768 + 1
    board = chess.Board()
    indices, values = features.active(board, "side_to_move")
    base, _ = plain.active(board, "side_to_move")
    # White to move, king on e1 (square 4): the block starts at 4 * 768.
    assert indices == [index + 4 * 768 for index in base]
    assert values == [1.0] * 32
    board.turn = chess.BLACK
    black, _ = features.active(board, "side_to_move")
    assert black == [index + 4 * 768 for index in plain.active(board, "side_to_move")[0]]
    white_view, white_values = features.active(board, "white")
    assert white_view[-1] == 64 * 768
    assert white_values[-1] == 0.0


def test_hybrid_features_put_the_pieces_in_the_king_bucket_and_the_terms_after_them():
    plain = SlottedPstFeatures()
    terms = PstTermsFeatures()
    board = chess.Board()
    sizes = {}
    for name, buckets in (("pst_hybrid", 1), ("pst_hybrid_k4", 4), ("pst_hybrid_k8", 8)):
        features = FEATURE_SETS[name]
        sizes[name] = features.size("side_to_move")
        assert sizes[name] == 768 * buckets + 22
        assert features.size("white") == sizes[name] + 1
        assert features.max_entries("side_to_move") == 54
        indices, values = features.active(board, "side_to_move")
        assert len(indices) == len(values) == 54
        pieces = plain.active(board, "side_to_move")[0]
        # King on e1: file group 2 of 4 (or group 1 of 2), rank group 0.
        bucket = {1: 0, 4: 2, 8: 4}[buckets]
        assert indices[:32] == [index + 768 * bucket for index in pieces]
        assert values[:32] == [1.0] * 32
        # The term entries equal those of linear_pst_terms, moved behind the buckets.
        expected = terms.active(board, "side_to_move")
        assert indices[32:] == list(range(768 * buckets, 768 * buckets + 22))
        assert values[32:] == expected[1][-22:]
    white_indices, white_values = FEATURE_SETS["pst_hybrid_k4"].active(board, "white")
    assert white_indices[-1] == 768 * 4 + 22
    assert white_values[-1] == 1.0
    board.turn = chess.BLACK
    # Black to move: the bucket is read on the colour-mirrored king square (e8 -> e1).
    black = FEATURE_SETS["pst_hybrid_k8"].active(board, "side_to_move")[0]
    assert max(black[:32]) < 768 * 8
    assert black[:32] == [index + 768 * 4 for index in plain.active(board, "side_to_move")[0]]


def test_hybrid_king_buckets_split_files_in_groups_and_ranks_at_the_third():
    features = FEATURE_SETS["pst_hybrid_k8"]
    assert features.bucket(chess.E1) == 4
    assert features.bucket(chess.E3) == 5
    assert features.bucket(chess.A2) == 0
    assert features.bucket(chess.H8) == 7
    assert FEATURE_SETS["pst_hybrid_k4"].bucket(chess.D1) == 0
    assert FEATURE_SETS["pst_hybrid_k4"].bucket(chess.E4) == 3
    assert FEATURE_SETS["pst_hybrid"].bucket(chess.E4) == 0
