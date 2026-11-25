"""Tests of the model cards."""

import json

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
