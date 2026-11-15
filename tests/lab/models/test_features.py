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
