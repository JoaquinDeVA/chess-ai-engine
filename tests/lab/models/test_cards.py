"""Tests of the model cards."""

import json
import re
import subprocess

from chess_lab.models.store import cards
from tests.conftest import ACCURACY


def test_update_merges_dictionaries_and_replaces_scalars(isolated_cards):
    cards.update_card("m", architecture="a", training={"epochs": 3, "dataset": "x"})
    card = cards.update_card("m", architecture="b", training={"epochs": 4})
    assert card["architecture"] == "b"
    assert card["training"] == {"epochs": 4, "dataset": "x"}
    assert json.loads(cards.card_path("m").read_text())["training"]["epochs"] == 4


def test_clear_measurements_keeps_only_the_identity(isolated_cards):
    cards.update_card(
        "m",
        status="current",
        architecture="a",
        features="f",
        parameters=3,
        size_kb=1.5,
        training={"effort": 1},
        accuracy=ACCURACY,
        cost={"us_per_eval": 1.0},
        ordering={"all": {"tau": 0.5}},
        games={"all": {"n": 3}},
    )
    cards.clear_measurements("m")
    assert cards.load_card("m") == {
        "name": "m", "status": "current", "architecture": "a", "features": "f",
        "parameters": 3, "size_kb": 1.5,
    }  # fmt: skip
    assert json.loads(cards.card_path("m").read_text()) == cards.load_card("m")


def test_clear_measurements_of_a_missing_card_does_nothing(isolated_cards):
    cards.clear_measurements("ghost")
    assert not cards.card_path("ghost").exists()


def test_missing_card_is_an_empty_skeleton(isolated_cards):
    assert cards.load_card("ghost") == {"name": "ghost"}


def test_current_commit_marks_a_dirty_tree_and_is_unknown_outside_git(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (tmp_path / "a.txt").write_text("a")
    git("add", "a.txt")
    git("commit", "-q", "-m", "first")
    monkeypatch.setattr(cards, "ROOT", tmp_path)
    clean = cards.current_commit()
    assert re.fullmatch(r"[0-9a-f]{7,}", clean)
    (tmp_path / "a.txt").write_text("changed")
    assert cards.current_commit() == clean + "+dirty"
    monkeypatch.setattr(cards, "ROOT", tmp_path / "nowhere")
    assert cards.current_commit() == "unknown"
