"""Tests of the sparse features of the networks, seen from the side to move."""

import chess
import pytest

from chess_lab.models.features import (
    _PAWN_CACHE,
    FEATURE_SETS,
    TERM_NAMES,
    HybridPstFeatures,
    SlottedPstFeatures,
    _pawn_terms,
    phase,
)
from tests.conftest import random_positions


def test_phase_is_one_at_start_and_zero_with_bare_kings():
    assert phase(chess.Board()) == pytest.approx(1.0)
    assert phase(chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")) == pytest.approx(0.0)


def term_values(board: chess.Board) -> dict[str, float]:
    """Mover's value of every classical term (middlegame plus endgame weight of its entries)."""
    features = HybridPstFeatures()
    indices, values = features.active(board)
    base = features.size() - 2 * len(TERM_NAMES)
    totals = dict.fromkeys(TERM_NAMES, 0.0)
    for index, value in zip(indices, values, strict=True):
        if index >= base:
            totals[TERM_NAMES[(index - base) // 2]] += value
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
    # The opponent's pair counts against the mover.
    assert term_values(chess.Board("2b1kb2/8/8/8/8/8/8/4K3 w - - 0 1"))["bishop_pair"] == -1.0


def test_terms_are_seen_from_the_side_to_move():
    board = chess.Board("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1")
    assert term_values(board)["bishop_pair"] == 1.0
    board.turn = chess.BLACK
    assert term_values(board)["bishop_pair"] == -1.0


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
            assert sorted(zip(*entries, strict=True)) == pytest.approx(
                sorted(zip(*mirrored, strict=True))
            )


def test_hybrid_features_put_the_pieces_in_the_king_bucket_and_the_terms_after_them():
    plain = SlottedPstFeatures()
    board = chess.Board()
    for name, buckets in (("pst_hybrid", 1), ("pst_hybrid_k4", 4), ("pst_hybrid_k8", 8)):
        features = FEATURE_SETS[name]
        assert features.size() == 768 * buckets + 22
        assert features.max_entries() == 54
        indices, values = features.active(board)
        assert len(indices) == len(values) == 54
        pieces = plain.active(board)[0]
        # King on e1: file group 2 of 4 (or group 1 of 2), rank group 0.
        bucket = {1: 0, 4: 2, 8: 4}[buckets]
        assert indices[:32] == [index + 768 * bucket for index in pieces]
        assert values[:32] == [1.0] * 32
        assert indices[32:] == list(range(768 * buckets, 768 * buckets + 22))
    board.turn = chess.BLACK
    # Black to move: the bucket is read on the colour-mirrored king square (e8 -> e1).
    black = FEATURE_SETS["pst_hybrid_k8"].active(board)[0]
    assert black[:32] == [index + 768 * 4 for index in plain.active(board)[0]]


def test_hybrid_king_buckets_split_files_in_groups_and_ranks_at_the_third():
    features = FEATURE_SETS["pst_hybrid_k8"]
    assert features.bucket(chess.E1) == 4
    assert features.bucket(chess.E3) == 5
    assert features.bucket(chess.A2) == 0
    assert features.bucket(chess.H8) == 7
    assert FEATURE_SETS["pst_hybrid_k4"].bucket(chess.D1) == 0
    assert FEATURE_SETS["pst_hybrid_k4"].bucket(chess.E4) == 3
    assert FEATURE_SETS["pst_hybrid"].bucket(chess.E4) == 0
