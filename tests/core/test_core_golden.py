"""The Rust core against the golden files made by the Python engine (tests/golden/)."""

import json

import pytest

from chess_engine import _core
from chess_engine.models import load_model
from chess_engine.models.native import h1_evaluator, model_evaluator
from tests.paths import GOLDEN, ROOT, SYZYGY

MODELS = ("linear_material", "linear_pst", "linear_pst_terms", "linear_king_bucket", "net4_pst")
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def golden(name):
    return json.loads((GOLDEN / name).read_text())


@pytest.fixture(scope="module")
def positions():
    return golden("positions.json")["positions"]


def evaluator_of(player):
    return h1_evaluator() if player == "h1" else model_evaluator(load_model(player))


# The linear feature sets are checked against the live Python code by tests/core/test_feature_parity.py;
# only the network's features keep a golden file.
@pytest.mark.parametrize("name", ("net4_pst",))
def test_features_match_golden(name, positions):
    expected = golden(f"features_{name}.json")
    evaluator = evaluator_of(name)
    assert len(expected["rows"]) == len(positions)
    for entry, (indices, values) in zip(positions, expected["rows"], strict=True):
        got_indices, got_values = evaluator.features(entry["fen"])
        assert got_indices == indices, entry["fen"]
        assert got_values == pytest.approx(values, abs=1e-12), entry["fen"]


def test_h1_features_match_golden(positions):
    expected = golden("features_h1.json")["rows"]
    for entry, row in zip(positions, expected, strict=True):
        assert _core.h1_features(entry["fen"]) == pytest.approx(row, abs=1e-12), entry["fen"]


@pytest.mark.parametrize("player", ("h1", *MODELS))
def test_evaluations_match_golden(player, positions):
    rows = golden(f"eval_{player}.json")["rows"]
    evaluator = evaluator_of(player)
    for entry, row in zip(positions, rows, strict=True):
        white, black = (
            evaluator.evaluate(entry["fen"], True),
            evaluator.evaluate(entry["fen"], False),
        )
        assert (white, black) == pytest.approx(tuple(row[-2:]), abs=1e-9), entry["fen"]


@pytest.mark.parametrize("player", ("h1", *MODELS))
def test_fixed_depth_searches_match_golden(player, positions):
    evaluator = evaluator_of(player)
    results = golden(f"search_{player}.json")["results"]
    assert len(results) > 200
    for case in results:
        entry = positions[case["index"]]
        searcher = _core.Searcher()
        value, move, nodes, prunings, max_ply = searcher.search_fixed(
            entry["root"], entry["moves"], evaluator, case["depth"], case["quiescence"]
        )
        label = f"{player} {entry['fen']} depth {case['depth']} q {case['quiescence']}"
        assert value == pytest.approx(case["value"], abs=1e-9), label
        assert move == case["move"], label
        assert (nodes, prunings, max_ply) == (case["nodes"], case["prunings"], case["max_ply"]), (
            label
        )


def test_terminal_matches_golden():
    for case in golden("terminal.json")["cases"]:
        assert _core.terminal(case["root"], case["moves"]) == (case["over"], case["winner"]), case[
            "fen"
        ]


def test_book_matches_golden():
    book = _core.Book.load(str(ROOT / "data" / "opening_book.json"))
    expected = golden("book.json")
    assert len(book) == expected["size"]
    for case in expected["cases"]:
        assert book.lookup(case["fen"]) == case["move"], case["fen"]


def test_tablebase_matches_golden():
    expected = golden("tablebase.json")
    tables = _core.Tablebase.open(str(SYZYGY))
    for case in expected["three_piece"]:
        assert tables.best_move(case["fen"]) == case["move"], case["fen"]


def test_the_incremental_network_gives_the_same_search():
    evaluator = evaluator_of("net4_pst")
    results = golden("search_net4_pst.json")["results"]
    positions = golden("positions.json")["positions"]
    checked = 0
    for case in results:
        if case["depth"] != 3 or case["quiescence"] != 4:
            continue
        entry = positions[case["index"]]
        value, move, nodes, _, _ = _core.Searcher().search_fixed(
            entry["root"], entry["moves"], evaluator, 3, 4, [], ["incremental"]
        )
        assert value == pytest.approx(case["value"], abs=1e-9), entry["fen"]
        assert abs(nodes - case["nodes"]) <= 0.01 * case["nodes"] + 1, entry["fen"]
        checked += move == case["move"]
    assert checked >= 0.9 * sum(c["depth"] == 3 and c["quiescence"] == 4 for c in results)
