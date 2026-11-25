"""Parity of the Rust core with the Python side: the dense reference of each network and the training graph."""

import chess
import numpy as np
import pytest

from chess_lab.models.definitions import (
    NeuralEvalModel,
    NnThreatsW64BothModel,
    NnW64BothModel,
    NnW128Ob8BothModel,
    NnW128Ob8H16BothModel,
)
from chess_lab.models.features import (
    KING_BASE,
    KING_BUCKETS,
    MATERIAL_BASE,
    PAWN_FILES_BASE,
    features_of,
)
from chess_lab.models.target import Representation
from tests.conftest import CTX_MODELS, mover_score, random_net, random_positions
from tests.lab.models.positions import frame


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
    w0, b0 = model.layers[0]
    rival = board.copy()
    rival.turn = not board.turn
    views = [ctx_table_rows(model, board), ctx_table_rows(model, rival)]
    h = np.concatenate([np.clip(w0[rows].sum(axis=0) + b0, 0.0, 1.0) for rows in views])
    return output_logit(model, h, board)


def reference_bucket(model: NeuralEvalModel, board: chess.Board) -> int:
    """The output unit of ``board``: its pieces on the board, four counts per bucket of eight."""
    buckets = model.layers[-1][0].shape[1]
    return (len(board.piece_map()) - 1) * buckets // 32


def output_logit(model: NeuralEvalModel, h: np.ndarray, board: chess.Board) -> float:
    """The output of ``board`` over both accumulators ``h``: linear, or its bucket's head."""
    bucket = reference_bucket(model, board)
    w1, b1 = model.layers[1]
    if len(model.layers) == 2:
        return float((h @ w1 + b1)[bucket])
    (w2, b2), head = model.layers[2], model.layers[2][0].shape[0]
    columns = slice(bucket * head, (bucket + 1) * head)
    hidden = np.clip(h @ w1[:, columns] + b1[columns], 0.0, 1.0)
    return float(hidden @ w2[:, bucket] + b2[bucket])


def dense_logit(model: NeuralEvalModel, board: chess.Board) -> float:
    """Reference forward pass on a dense 768- (or 769-) vector, same weights as ``model``."""
    if model.feature_set == "pst_ctx":
        return ctx_logit(model, board)
    indices, values = features_of(model).active(board)
    x = np.zeros(features_of(model).size())
    for index, value in zip(indices, values, strict=True):
        x[index] += value
    w0, b0 = model.layers[0]
    h = np.clip(x @ w0 + b0, 0.0, 1.0)
    # the rival's view: every piece in the other block of its square; the dense entries have no
    # rival view
    rival = np.zeros_like(x)
    for index, value in zip(indices, values, strict=True):
        if index < 768:
            rival[(index + 384) % 768] += value
    h = np.concatenate([h, np.clip(rival @ w0 + b0, 0.0, 1.0)])
    return output_logit(model, h, board)


BOTH_MODELS = [NnW64BothModel, NnThreatsW64BothModel, NnW128Ob8BothModel, NnW128Ob8H16BothModel]


@pytest.mark.parametrize("model_class", [*BOTH_MODELS, *CTX_MODELS])
def test_core_score_matches_a_dense_numpy_reference(model_class):
    model = random_net(seed=6, model_class=model_class)
    evaluator = model.evaluator()
    for board in random_positions(10, seed=8):
        assert evaluator.evaluate(board.fen(), board.turn) == pytest.approx(
            mover_score(dense_logit(model, board))
        )


def keras_logits(model: NeuralEvalModel, boards: list[chess.Board]) -> np.ndarray:
    """The logits of the training graph of ``model`` for ``boards``."""
    from chess_lab.training.epochs import encode_frame
    from chess_lab.training.neural import _assign, _keras_network, _padded, _training_keras

    features = features_of(model)
    width = features.max_entries()
    network, parts = _keras_network(
        _training_keras(),
        [w.shape for w, _ in model.layers],
        width,
        0.0,
        model.families,
        model.feature_set == "pst_ctx",
    )
    _assign(parts, model.layers)
    index, value, _ = _padded(encode_frame(features, Representation(), frame(boards)), width)
    return np.asarray(network.predict([index, value], verbose=0)).reshape(-1)


@pytest.mark.parametrize("model_class", [*BOTH_MODELS, *CTX_MODELS])
def test_the_training_graph_computes_what_the_core_does(model_class):
    model = random_net(seed=5, model_class=model_class)
    boards = random_positions(40, seed=9)
    for board, logit in zip(boards, keras_logits(model, boards), strict=True):
        assert logit == pytest.approx(dense_logit(model, board), abs=1e-4)


def one_board_per_bucket() -> list[chess.Board]:
    """The start position with pieces removed: 3, 7, ..., 31 pieces, one in each of 8 buckets."""
    boards = []
    for pieces in range(3, 33, 4):
        board = chess.Board()
        for square in [s for s, p in board.piece_map().items() if p.piece_type != chess.KING][
            : 32 - pieces
        ]:
            board.remove_piece_at(square)
        boards.append(board)
    return boards


@pytest.mark.parametrize("model_class", [NnW128Ob8BothModel, NnW128Ob8H16BothModel])
def test_every_output_bucket_is_read_alike_by_the_core_and_the_training_graph(model_class):
    model = random_net(seed=4, model_class=model_class)
    boards = one_board_per_bucket()
    assert [reference_bucket(model, b) for b in boards] == list(range(8))
    evaluator = model.evaluator()
    for board, logit in zip(boards, keras_logits(model, boards), strict=True):
        assert logit == pytest.approx(dense_logit(model, board), abs=1e-4)
        assert evaluator.evaluate(board.fen(), board.turn) == pytest.approx(mover_score(logit))


def boards_by_piece_count() -> list[chess.Board]:
    """From the start position, one board for each count of pieces from 32 down to the two kings."""
    boards, board = [], chess.Board()
    for _ in range(30):
        boards.append(board.copy())
        victim = next(s for s, p in board.piece_map().items() if p.piece_type != chess.KING)
        board.remove_piece_at(victim)
    boards.append(board.copy())
    return boards


@pytest.mark.parametrize("buckets", [1, 2, 4, 16, 32])
def test_the_core_picks_the_output_unit_by_pieces_for_any_number_of_buckets(buckets):
    model_class = type("Buckets", (NnW64BothModel,), {"output_buckets": buckets})
    model = random_net(seed=2, model_class=model_class)
    evaluator = model.evaluator()
    boards = boards_by_piece_count()
    assert len({reference_bucket(model, b) for b in boards}) == min(buckets, len(boards))
    for board in boards:
        assert evaluator.evaluate(board.fen(), board.turn) == pytest.approx(
            mover_score(dense_logit(model, board))
        )


#: Rare cases of every input rule: the bare ones are the ones a random game seldom reaches.
EDGE_FENS = [
    "8/P6k/8/8/8/8/6p1/K7 w - - 0 1",  # a promotion ahead for each side
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",  # en passant available
    "4k3/8/8/8/8/8/8/QQQQKQQQ w - - 0 1",  # seven queens on the board
    "7k/8/8/8/8/8/QQQQQQQQ/QQQQKQQQ b - - 0 1",  # fifteen queens, past the material cap
    "4k3/8/8/8/8/8/8/NNN1K3 b - - 0 1",  # three knights, black to move
    "k7/8/8/8/8/8/8/7K w - - 0 1",  # kings in the corners
    "4k3/8/8/8/8/8/8/4K3 b - - 0 1",  # bare kings, black to move
    "4k3/pppp4/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",  # columns with 0, 1 and 2 pawns
    "4k3/8/8/8/PPPP4/PPPP4/PPPP4/4K3 w - - 0 1",  # columns with 3 pawns
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # castling rights
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",  # half of them, black to move
    "7k/8/8/8/8/8/8/K6R b - - 0 1",  # black in check, the king on the edge
]


@pytest.mark.parametrize("model_class", [*BOTH_MODELS, *CTX_MODELS])
def test_the_core_agrees_with_the_reference_on_the_rare_cases_of_every_input_rule(model_class):
    model = random_net(seed=7, model_class=model_class)
    evaluator = model.evaluator()
    for fen in EDGE_FENS:
        board = chess.Board(fen)
        assert evaluator.evaluate(fen, board.turn) == pytest.approx(
            mover_score(dense_logit(model, board)), abs=1e-6
        ), fen


@pytest.mark.parametrize("model_class", [NnW64BothModel, CTX_MODELS[-1]])
def test_the_two_sides_score_a_position_as_complements(model_class):
    model = random_net(seed=3, model_class=model_class)
    evaluator = model.evaluator()
    for board in random_positions(20, seed=5):
        white = evaluator.evaluate(board.fen(), True)
        black = evaluator.evaluate(board.fen(), False)
        assert white + black == pytest.approx(1.0, abs=1e-6)
