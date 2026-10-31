"""Tests of src/chess_lab/scripts/data/build_opening_book.py with a fake evaluation table (no shards)."""

import importlib.util
from pathlib import Path

import chess
import numpy as np
import pytest


@pytest.fixture(scope="module")
def script():
    path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "chess_lab"
        / "scripts"
        / "data"
        / "build_opening_book.py"
    )
    spec = importlib.util.spec_from_file_location("build_opening_book", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def key_after(script, *ucis):
    board = chess.Board()
    for uci in ucis:
        board.push_uci(uci)
    return script.position_key(board)


def table(script, entries):
    """Lookup over ``{(moves...): white score}``."""
    scores = {key_after(script, *moves): score for moves, score in entries.items()}
    return scores.get


def test_white_scores_turn_mates_into_large_signed_scores(script):
    cp = np.array([35.0, np.nan, np.nan])
    mate = np.array([np.nan, 3.0, -2.0])
    result = script.white_scores(cp, mate)
    assert result[0] == 35.0
    assert result[1] == script.MATE_SCORE - 3 and result[2] == -(script.MATE_SCORE - 2)


def test_each_side_takes_the_move_that_is_best_for_itself(script):
    lookup = table(
        script,
        {("e2e4",): 30, ("d2d4",): 25, ("e2e4", "e7e5"): 20, ("e2e4", "c7c5"): 35},
    )
    book = script.build_book(lookup, plies=2, margin=30, top=3, eval_limit=150, cap_per_ply=100)
    assert book[key_after(script)] == "e2e4"  # White prefers the higher score
    assert book[key_after(script, "e2e4")] == "e7e5"  # Black prefers the lower one


def test_only_moves_within_the_margin_are_followed(script):
    lookup = table(
        script,
        {("e2e4",): 30, ("c2c4",): 5, ("e2e4", "e7e5"): 25, ("c2c4", "e7e5"): 5},
    )
    book = script.build_book(lookup, plies=2, margin=10, top=3, eval_limit=150, cap_per_ply=100)
    assert key_after(script, "e2e4") in book
    assert key_after(script, "c2c4") not in book


def test_lines_beyond_the_evaluation_limit_are_not_followed(script):
    lookup = table(script, {("e2e4",): 30, ("d2d4",): 400, ("d2d4", "d7d5"): 390})
    book = script.build_book(lookup, plies=2, margin=1000, top=3, eval_limit=150, cap_per_ply=100)
    assert key_after(script, "d2d4") not in book


def test_a_level_keeps_the_most_balanced_positions_up_to_the_cap(script):
    lookup = table(
        script,
        {("e2e4",): 40, ("d2d4",): 10, ("e2e4", "e7e5"): 30, ("d2d4", "d7d5"): 5},
    )
    book = script.build_book(lookup, plies=2, margin=100, top=3, eval_limit=150, cap_per_ply=1)
    assert key_after(script, "d2d4") in book and key_after(script, "e2e4") not in book


def test_positions_without_evaluated_moves_or_with_only_losing_moves_get_no_entry(script):
    book = script.build_book(lambda key: None, 3, 30, 3, 150, 100)
    assert book == {}
    losing = table(script, {("e2e4",): -300, ("d2d4",): -400})
    assert script.build_book(losing, 2, 30, 3, 150, 100) == {}


def test_a_transposition_is_expanded_once(script):
    lookups = []
    base = table(
        script,
        {
            ("e2e4",): 20,
            ("d2d4",): 20,
            ("e2e4", "d7d5"): 20,
            ("d2d4", "d7d5"): 20,
            ("e2e4", "d7d5", "d2d4"): 20,
            ("d2d4", "d7d5", "e2e4"): 20,
            ("e2e4", "d7d5", "d2d4", "g8f6"): 20,
        },
    )

    def counting(key):
        lookups.append(key)
        return base(key)

    book = script.build_book(counting, plies=4, margin=100, top=5, eval_limit=150, cap_per_ply=100)
    transposed = key_after(script, "e2e4", "d7d5", "d2d4")
    assert transposed == key_after(script, "d2d4", "d7d5", "e2e4")
    assert transposed in book
    # the children of the transposed position were looked up once, not once per route
    assert lookups.count(key_after(script, "e2e4", "d7d5", "d2d4", "g8f6")) == 1


def test_lookup_finds_indexed_keys_and_misses_others(script):
    order = sorted(["a b c d", "e f g h"], key=hash)
    hashes = np.array([hash(k) for k in order], dtype=np.int64)
    scores = np.array([1.0, 2.0], dtype=np.float32)
    lookup = script.make_lookup(hashes, scores)
    assert lookup(order[0]) == 1.0 and lookup(order[1]) == 2.0
    assert lookup("not indexed") is None
