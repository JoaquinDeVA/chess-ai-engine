"""Tests of what the opening-book scripts share: keys, the book file and JSON lines."""

import json

import chess

from chess_lab.datasets.book import (
    append_jsonl,
    fen_key,
    load_book,
    position_key,
    read_jsonl,
)
from chess_lab.datasets.check_opening_book import load_cache


def test_the_key_drops_the_clocks_so_a_transposition_hits_the_same_entry():
    board = chess.Board()
    assert (
        position_key(board)
        == fen_key(board.fen())
        == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
    )
    assert fen_key(board.fen()) == fen_key(" ".join(board.fen().split()[:4]))


def test_a_book_is_read_with_its_keys_reduced_to_four_fields(tmp_path):
    path = tmp_path / "book.json"
    path.write_text(json.dumps({chess.Board().fen(): "e2e4"}), encoding="utf-8")
    assert load_book(path) == {position_key(chess.Board()): "e2e4"}


def test_a_cut_last_line_is_skipped_and_a_missing_file_is_empty(tmp_path):
    path = tmp_path / "cache.jsonl"
    assert read_jsonl(path) == []
    with path.open("a", encoding="utf-8") as handle:
        append_jsonl(handle, {"key": "a", "answer": 1})
        handle.write('{"key": "b", "ans')  # a write cut by a Ctrl-C
    assert read_jsonl(path) == [{"key": "a", "answer": 1}]
    assert load_cache(path) == {"a": 1}
