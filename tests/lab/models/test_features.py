"""Tests of the sparse features of the networks, seen from the side to move."""

import chess
import pytest

from chess_lab.models.features import FEATURE_SETS, SlottedPstFeatures
from tests.conftest import random_positions


def test_slotted_features_are_one_hot_and_mover_relative():
    features = SlottedPstFeatures()
    indices, values = features.active(chess.Board())
    assert len(indices) == 32
    assert values == [1.0] * 32
    assert features.size() == 768
    assert features.max_entries() == 32
    assert sorted(indices) == sorted(features.active(chess.Board().mirror())[0])
    for board in random_positions(200, seed=21):
        indices, _ = features.active(board)
        assert sorted(indices) == sorted(features.active(board.mirror())[0])
        assert len(indices) <= features.max_entries()


def test_every_feature_set_is_mirror_symmetric():
    for features in FEATURE_SETS.values():
        for board in random_positions(30, seed=7):
            entries = features.active(board)
            mirrored = features.active(board.mirror())
            got = sorted(zip(*entries, strict=True))
            want = sorted(zip(*mirrored, strict=True))
            assert [i for i, _ in got] == [i for i, _ in want]
            assert [v for _, v in got] == pytest.approx([v for _, v in want])


@pytest.mark.parametrize(
    ("fen", "expected"),
    [
        (chess.STARTING_FEN, [0, 0, 0, 0, 0]),
        # the pawn wins the undefended queen (9); the queen wins the undefended pawn (1)
        ("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", [1, 1 / 9, 0.25, 0.25, 0]),
        # the pawn takes a rook defended by a pawn: 5 - 1
        ("4k3/8/4p3/3r4/4P3/8/8/4K3 w - - 0 1", [4 / 9, 0, 0.25, 0, 0]),
        # the king takes the undefended rook (5) and is in check
        ("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", [5 / 9, 0, 0.25, 0, 1]),
    ],
)
def test_threat_inputs_follow_the_hand_checked_positions(fen, expected):
    features = FEATURE_SETS["pst_threats"]
    indices, values = features.active(chess.Board(fen))
    assert indices[-5:] == list(range(768, 773))
    assert values[-5:] == pytest.approx(expected)


def test_threat_set_has_the_pieces_and_five_dense_entries():
    features = FEATURE_SETS["pst_threats"]
    assert features.size() == 768 + 5
    assert features.max_entries() == 37


def _ctx_extras(fen):
    """Entries of ``pst_ctx`` past the pieces, split by block."""
    features = FEATURE_SETS["pst_ctx"]
    indices, values = features.active(chess.Board(fen))
    assert values == [1.0] * len(indices)
    extra = indices[len([i for i in indices if i < 768]) :]
    return (
        [i - 768 for i in extra if 768 <= i < 798],
        [i - 798 for i in extra if 798 <= i < 830],
        [i - 830 for i in extra if i >= 830],
    )


def test_ctx_set_sizes():
    features = FEATURE_SETS["pst_ctx"]
    assert features.size() == 768 + 30 + 32 + 8
    assert features.max_entries() == 60
    for board in random_positions(200, seed=3):
        assert len(features.active(board)[0]) <= features.max_entries()


def test_ctx_entries_of_the_start_position():
    material, pawns, buckets = _ctx_extras(chess.STARTING_FEN)
    # per side: 8 pawns (7), 2 knights (9), 2 bishops (11), 2 rooks (13), 1 queen (14)
    side = [7, 9, 11, 13, 14]
    assert material == side + [i + 15 for i in side]
    # every file has one pawn: entry 2 * file; the rival's block starts at 16
    assert pawns == [2 * f for f in range(8)] + [16 + 2 * f for f in range(8)]
    # both kings at home (bucket 2), mover's then rival's
    assert buckets == [2, 4 + 2]


def test_ctx_counts_files_and_buckets_after_castling_and_capture():
    # White castled short with six pawns (doubled on the c-file) and a rook; Black has the king
    # on e8 and pawns on d7 and e7; white to move
    fen = "4k3/3pp3/8/8/8/2P5/P1P2PPP/5RK1 w - - 0 1"
    material, pawns, buckets = _ctx_extras(fen)
    # mover: 6 pawns -> index 5, 1 rook -> 12 (first rook entry); rival: 2 pawns -> 15 + 1
    assert material == [5, 12, 15 + 1]
    # mover: a (1 pawn) -> 0, c (2 pawns) -> 5, f -> 10, g -> 12, h -> 14; rival: d -> 16 + 6,
    # e -> 16 + 8
    assert pawns == [0, 5, 10, 12, 14, 22, 24]
    assert buckets == [0, 4 + 2]


def test_ctx_king_bucket_is_seen_from_the_home_rank():
    # Black king castled long, seen from its home rank it is on c1: bucket 1; white king on h4
    fen = "2kr4/8/8/8/7K/8/8/8 w - - 0 1"
    _, _, buckets = _ctx_extras(fen)
    assert buckets == [3, 4 + 1]
