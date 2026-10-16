"""Tests of the Prolog tactical-pattern knowledge base (skipped without SWI-Prolog)."""

from pathlib import Path

import pytest

pyswip = pytest.importorskip("pyswip")
try:
    pyswip.Prolog()
except Exception:  # SwiPrologNotFoundError when the interpreter is missing
    pytest.skip("SWI-Prolog is not installed", allow_module_level=True)

KNOWLEDGE_BASE = Path(__file__).resolve().parent.parent / "knowledge" / "chess_patterns.pl"


@pytest.fixture(scope="module")
def prolog():
    """Prolog engine with the pattern knowledge base loaded."""
    pl = pyswip.Prolog()
    pl.consult(str(KNOWLEDGE_BASE))

    # Declare the dynamic predicate used by the tests without redefining static ones
    try:
        pl.assertz("move(dummy, dummy, dummy, dummy)")
    except ValueError:
        pass  # already defined or protected

    yield pl

    pl.retractall("piece(_,_,_,_)")
    pl.retractall("move(_,_,_,_)")


def reset_pieces(pl):
    """Remove every ``piece/4`` fact."""
    pl.retractall("piece(_,_,_,_)")


# ----------------------------------------------------------------------
# Pattern tests
# ----------------------------------------------------------------------
def test_absolute_pin(prolog):
    """Rook pinning knight to king"""
    prolog.assertz("piece(r, white, 4, 3)")
    prolog.assertz("piece(n, black, 6, 4)")
    prolog.assertz("piece(k, black, 8, 4)")
    prolog.assertz("move(4,3,4,4)")

    result = list(prolog.query("move_creates_absolute_pin(4,3,4,4,Score)."))
    assert result, "No se detectó el pin absoluto"
    assert result[0]["Score"] > 0

    reset_pieces(prolog)


def test_relative_pin(prolog):
    """Rook pinning knight to queen"""
    prolog.assertz("piece(r, white, 4, 3)")
    prolog.assertz("piece(n, black, 6, 4)")
    prolog.assertz("piece(q, black, 8, 4)")
    prolog.assertz("move(4,3,4,4)")

    result = list(prolog.query("move_creates_relative_pin(4,3,4,4,Score)."))
    assert result and result[0]["Score"] > 0
    reset_pieces(prolog)


def test_fork(prolog):
    """Knight forking queen and rook"""
    prolog.assertz("piece(n, white, 4, 4)")
    prolog.assertz("piece(q, black, 7, 4)")
    prolog.assertz("piece(r, black, 4, 7)")
    prolog.assertz("move(4,4,5,5)")

    result = list(prolog.query("move_creates_fork(4,4,5,5,Score)."))
    assert result and result[0]["Score"] > 0
    reset_pieces(prolog)


def test_skewer(prolog):
    """Rook skewering queen and bishop"""
    prolog.assertz("piece(r, white, 4, 4)")
    prolog.assertz("piece(q, black, 6, 6)")
    prolog.assertz("piece(b, black, 8, 8)")

    result = list(prolog.query("move_creates_skewer(4,4,5,5,Score)."))
    assert result and result[0]["Score"] > 0
    reset_pieces(prolog)


def test_capture(prolog):
    """Queen captures rook"""
    prolog.assertz("piece(q, white, 4, 4)")
    prolog.assertz("piece(r, black, 5, 5)")

    result = list(prolog.query("move_creates_capture(4,4,5,5,Score)."))
    assert result and result[0]["Score"] == (5 * 10) - 9
    reset_pieces(prolog)


def test_promotion_white(prolog):
    """White pawn promotes"""
    prolog.assertz("piece(p, white, 7, 1)")
    prolog.assertz("move(7,1,8,1)")
    result = list(prolog.query("move_creates_promotion(7,1,8,1)."))
    assert result
    reset_pieces(prolog)


def test_promotion_black(prolog):
    """Black pawn promotes"""
    prolog.assertz("piece(p, black, 2, 1)")
    prolog.assertz("move(2,1,1,1)")
    result = list(prolog.query("move_creates_promotion(2,1,1,1)."))
    assert result
    reset_pieces(prolog)


def test_no_capture_same_color(prolog):
    """Same-color pieces -> no capture"""
    prolog.assertz("piece(q, white, 1, 4)")
    prolog.assertz("piece(r, white, 1, 5)")
    result = list(prolog.query("move_creates_capture(1,4,1,5,_)."))
    assert result == []
    reset_pieces(prolog)


def test_no_piece_at_source(prolog):
    """No attacker on source square"""
    result = list(prolog.query("move_creates_capture(4,4,3,1,_)."))
    assert result == []


def test_no_piece_at_target(prolog):
    """Empty target square"""
    prolog.assertz("piece(p, white, 2, 1)")
    result = list(prolog.query("move_creates_capture(2,1,4,1,_)."))
    assert result == []
    reset_pieces(prolog)
