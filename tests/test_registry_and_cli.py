"""Bot discovery and the command-line interface."""

import subprocess
import sys
from pathlib import Path

import pytest

from chess_engine.bots.bot_registry import BotRegistry

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def registry() -> BotRegistry:
    return BotRegistry()


def test_registry_finds_every_playable_bot(registry):
    names = set(registry.list_available_bots())
    assert names == {"HumanPlayer", "IterativeHeuristicBot", "IterativeModelBookBot"}


def test_shared_base_classes_are_not_registered_as_bots(registry):
    assert not {"IterativeDeepeningBot", "ChessBot"} & set(registry.list_available_bots())


def test_registry_works_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "IterativeHeuristicBot" in BotRegistry().list_available_bots()


def test_get_bot_caches_plain_instances_and_builds_fresh_ones_for_options(registry):
    assert registry.get_bot("IterativeHeuristicBot") is registry.get_bot("IterativeHeuristicBot")
    deep = registry.get_bot("IterativeHeuristicBot", max_depth=4)
    assert deep.max_depth == 4 and deep is not registry.get_bot("IterativeHeuristicBot")


def test_unknown_bot_and_bad_options_return_none(registry):
    assert registry.get_bot("Nope") is None
    assert registry.get_bot("IterativeModelBookBot", model="no_such_model") is None


def test_tournament_ready_bots_exclude_humans(registry):
    ready = registry.get_tournament_ready_bots()
    assert "HumanPlayer" not in ready and "IterativeHeuristicBot" in ready


def run_cli(*args, timeout=180):
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cli_rejects_unknown_bots_and_bad_positions():
    assert run_cli("--player1", "Nope").returncode == 1
    assert (
        run_cli(
            "--player1",
            "IterativeHeuristicBot",
            "--player2",
            "IterativeHeuristicBot",
            "--position",
            "garbage",
        ).returncode
        == 1
    )


def test_cli_passes_model_and_depth_only_to_bots_that_accept_them(registry):
    import argparse

    from main import bot_options

    args = argparse.Namespace(model="linear_material", depth=3, endgame_db=None, opening_book=None)
    assert bot_options(registry, "IterativeModelBookBot", args) == {
        "model": "linear_material",
        "max_depth": 3,
    }
    assert bot_options(registry, "IterativeHeuristicBot", args) == {"max_depth": 3}
    assert bot_options(registry, "HumanPlayer", args) == {}


def test_cli_passes_the_endgame_tables_only_to_bots_that_search(registry):
    import argparse
    from pathlib import Path

    from chess_engine.knowledge.endgame_tablebase import EndgameTablebase
    from main import bot_options

    tables = Path(__file__).resolve().parent / "data" / "syzygy"
    args = argparse.Namespace(model=None, depth=None, endgame_db=str(tables), opening_book=None)
    options = bot_options(registry, "IterativeHeuristicBot", args)
    assert isinstance(options["tablebase"], EndgameTablebase)
    options["tablebase"].close()
    assert bot_options(registry, "HumanPlayer", args) == {}


def test_opening_book_option_reaches_the_search_bots(registry, tmp_path):
    import argparse

    from chess_engine.knowledge.opening_book import OpeningBook
    from main import bot_options

    book = tmp_path / "book.json"
    book.write_text('{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": "e2e4"}')
    args = argparse.Namespace(model=None, depth=None, endgame_db=None, opening_book=str(book))
    options = bot_options(registry, "IterativeHeuristicBot", args)
    assert isinstance(options["opening_book"], OpeningBook) and len(options["opening_book"]) == 1
    assert bot_options(registry, "HumanPlayer", args) == {}
