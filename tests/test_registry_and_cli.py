"""Bot discovery and the command-line interface."""

import subprocess
import sys
from pathlib import Path

import pytest

from bots.bot_registry import BotRegistry
from tests.conftest import MATE_IN_ONE

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def registry() -> BotRegistry:
    return BotRegistry()


def test_registry_finds_every_playable_bot(registry):
    names = set(registry.list_available_bots())
    assert {
        "RandomBot",
        "HumanPlayer",
        "HeuristicAlphaBetaBot",
        "AdvancedHeuristicAlphaBetaBot",
        "MLBot",
        "AdvancedMLBot",
        "TacticalAlphaBetaBot",
        "NonTacticalAlphaBetaBot",
    } <= names


def test_shared_base_classes_are_not_registered_as_bots(registry):
    assert not {"FixedDepthBot", "IterativeDeepeningBot", "ChessBot"} & set(
        registry.list_available_bots()
    )


def test_registry_works_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "RandomBot" in BotRegistry().list_available_bots()


def test_get_bot_caches_plain_instances_and_builds_fresh_ones_for_options(registry):
    assert registry.get_bot("HeuristicAlphaBetaBot") is registry.get_bot("HeuristicAlphaBetaBot")
    deep = registry.get_bot("HeuristicAlphaBetaBot", max_depth=4)
    assert deep.max_depth == 4 and deep is not registry.get_bot("HeuristicAlphaBetaBot")


def test_unknown_bot_and_bad_options_return_none(registry):
    assert registry.get_bot("Nope") is None
    assert registry.get_bot("MLBot", model="no_such_model") is None


def test_tournament_ready_bots_exclude_humans(registry):
    ready = registry.get_tournament_ready_bots()
    assert "HumanPlayer" not in ready and "RandomBot" in ready


def run_cli(*args, timeout=180):
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cli_lists_bots():
    result = run_cli("--list-bots")
    assert result.returncode == 0
    assert "RandomBot" in result.stdout and "bots available" in result.stdout


def test_cli_plays_a_game_to_the_end_and_exports_the_pgn(tmp_path):
    pgn = tmp_path / "game.pgn"
    result = run_cli(
        "--player1",
        "HeuristicAlphaBetaBot",
        "--player2",
        "RandomBot",
        "--depth",
        "2",
        "--position",
        MATE_IN_ONE,
        "--export",
        str(pgn),
    )
    assert result.returncode == 0, result.stderr
    assert "Winner: white" in result.stdout
    assert "Ra8#" in pgn.read_text()


def test_cli_rejects_unknown_bots_and_bad_positions():
    assert run_cli("--player1", "Nope").returncode == 1
    assert (
        run_cli(
            "--player1", "RandomBot", "--player2", "RandomBot", "--position", "garbage"
        ).returncode
        == 1
    )


def test_cli_passes_model_and_depth_only_to_bots_that_accept_them(registry):
    import argparse

    from main import bot_options

    args = argparse.Namespace(model="linear_baseline", depth=3)
    assert bot_options(registry, "MLBot", args) == {"model": "linear_baseline", "max_depth": 3}
    assert bot_options(registry, "HeuristicAlphaBetaBot", args) == {"max_depth": 3}
    assert bot_options(registry, "RandomBot", args) == {}
