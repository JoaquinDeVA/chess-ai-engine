"""Prolog tactical-pattern detector (skipped when SWI-Prolog is not installed)."""

import chess
import pytest

pytest.importorskip("pyswip")
try:
    from chess_engine.knowledge.pattern_detector import PrologPatternDetector

    PrologPatternDetector()
except RuntimeError:
    pytest.skip("SWI-Prolog is not installed", allow_module_level=True)


@pytest.fixture(scope="module")
def detector():
    return PrologPatternDetector()


def test_facts_describe_the_position(detector):
    facts = detector.board_to_facts(chess.Board())
    assert "piece(k, white, 1, 5)" in facts and "to_move(white)" in facts
    assert (
        "can_castle(black, queenside)" in facts
        and len([f for f in facts if f.startswith("piece")]) == 32
    )


def test_en_passant_square_is_reported(detector):
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert "en_passant(6, 4)" in detector.board_to_facts(board)


def test_a_capture_of_the_queen_scores_higher_than_a_quiet_move(detector):
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    moves = list(board.legal_moves)
    patterns = detector.detect_move_patterns(board, moves)
    capture = chess.Move.from_uci("e4d5")
    quiet = chess.Move.from_uci("e1e2")
    assert patterns[capture]["mvv_lva_score"] > 0 == patterns[quiet]["mvv_lva_score"]


def test_a_knight_fork_is_detected(detector):
    board = chess.Board("2r1k3/8/8/1N6/8/8/8/4K3 w - - 0 1")  # Nd6+ forks king and rook
    moves = list(board.legal_moves)
    patterns = detector.detect_move_patterns(board, moves)
    assert patterns[chess.Move.from_uci("b5d6")]["fork_score"] > 0


def test_promotions_are_flagged(detector):
    board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    patterns = detector.detect_move_patterns(board, list(board.legal_moves))
    assert any(p["creates_promotion"] for move, p in patterns.items() if move.promotion)
