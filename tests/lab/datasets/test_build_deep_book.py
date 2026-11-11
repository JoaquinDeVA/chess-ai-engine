"""Tests of src/chess_lab/datasets/build_deep_book.py with a fake engine (no Stockfish)."""

import json
import math

import chess
import pytest

from chess_lab.datasets import build_deep_book as deep
from chess_lab.datasets.build_opening_book import position_key


def fake_engine(scores=(0.0, -20.0, -200.0)):
    """Analysis giving the first legal moves (in sorted UCI order) the given scores."""
    calls = []

    def analyse(board):
        calls.append(position_key(board))
        moves = sorted(m.uci() for m in board.legal_moves)
        return list(zip(moves, scores, strict=False))

    analyse.calls = calls
    return analyse


def test_reply_weights_drop_lines_beyond_the_margin_and_sum_to_one():
    weights = deep.reply_weights([("a", 50.0), ("b", 20.0), ("c", -100.0)], 80.0, 30.0)
    assert [uci for uci, _ in weights] == ["a", "b"]
    assert sum(w for _, w in weights) == pytest.approx(1.0)
    assert weights[1][1] / weights[0][1] == pytest.approx(math.exp(-1))


def test_only_the_opponent_spreads_its_probability():
    board = chess.Board()
    lines = [("e2e4", 30.0), ("d2d4", 0.0)]
    kids = dict((c.peek().uci(), r) for c, r in deep.children(board, lines, (1.0, 0.5), 80, 30))
    w = 1 / (1 + math.exp(-1))
    # Blancas mueven: con el motor de blancas solo se juega la mejor; con el de negras, ambas.
    assert kids["e2e4"] == pytest.approx((1.0, 0.5 * w))
    assert kids["d2d4"] == pytest.approx((0.0, 0.5 * (1 - w)))


def test_a_forced_line_goes_as_deep_as_the_budget():
    def pawn_push(board):
        # Una sola línea, siempre de peón: no repite posiciones.
        pawns = sorted(
            m.uci() for m in board.legal_moves if board.piece_type_at(m.from_square) == chess.PAWN
        )
        return [(pawns[0], 0.0)]

    book = deep.build_book(pawn_push, {}, budget=12)
    board = chess.Board()
    while position_key(board) in book:
        board.push_uci(book[position_key(board)])
    assert len(book) == 12 and board.ply() == 12


def test_the_book_holds_the_best_legal_move_of_each_analysed_position():
    book = deep.build_book(fake_engine(), {}, budget=40, workers=3)
    assert len(book) == 40
    start = position_key(chess.Board())
    assert book[start] == sorted(m.uci() for m in chess.Board().legal_moves)[0]


def test_a_second_run_takes_every_analysis_from_the_cache(tmp_path):
    cache, saved = {}, []
    first = deep.build_book(
        fake_engine(), cache, budget=30, on_analysed=lambda k, v: saved.append(k)
    )
    assert len(saved) == 30
    path = tmp_path / "cache.jsonl"
    path.write_text(
        "".join(json.dumps({"fen": k, "lines": cache[k]}) + "\n" for k in saved) + '{"fen": "cut',
        encoding="utf-8",
    )
    again = fake_engine()
    assert deep.build_book(again, deep.load_cache(path), budget=30) == first
    assert again.calls == []
