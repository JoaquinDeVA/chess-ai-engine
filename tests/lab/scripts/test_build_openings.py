"""Tests of src/chess_lab/scripts/build_openings.py with a fake engine (no Stockfish)."""

import importlib.util

import chess
import chess.engine
import pytest

from tests.paths import ROOT


@pytest.fixture(scope="module")
def script():
    path = ROOT / "src" / "chess_lab" / "scripts" / "build_openings.py"
    spec = importlib.util.spec_from_file_location("build_openings", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEngine:
    """Scores a position by its count of white pieces, in centipawns above 16."""

    def analyse(self, board, limit):
        pieces = len(board.pieces(chess.PAWN, chess.WHITE)) + 8
        return {"score": chess.engine.PovScore(chess.engine.Cp((pieces - 16) * 100), chess.WHITE)}


def test_playouts_are_distinct_8_to_10_plies_deep_and_repeat_with_the_seed(script):
    fens = script.playouts(30, seed=3)
    assert len(set(fens)) == 30
    assert all(script.MIN_PLY <= chess.Board(fen).ply() <= script.MAX_PLY for fen in fens)
    assert fens == script.playouts(30, seed=3)
    assert fens != script.playouts(30, seed=4)


def test_balanced_keeps_only_scores_inside_the_limit(script):
    even = chess.STARTING_FEN
    pawn_down = chess.Board()
    pawn_down.remove_piece_at(chess.E2)
    kept = script.balanced([even, pawn_down.fen()], FakeEngine(), depth=1, limit=0.5)
    assert kept == [even]
