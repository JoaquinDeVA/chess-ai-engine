"""Bot discovery."""

import pytest

from chess_engine.bots.bot_registry import BotRegistry


@pytest.fixture(scope="module")
def registry() -> BotRegistry:
    return BotRegistry()


def test_registry_finds_every_playable_bot(registry):
    names = set(registry.list_available_bots())
    assert names == {"IterativeHeuristicBot", "IterativeModelBookBot"}


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
