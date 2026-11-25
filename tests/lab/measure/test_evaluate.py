"""Tests of the evaluation CLI on the game-position set."""

import numpy as np
import pandas as pd
import pytest

from chess_lab.measure import evaluate as evaluate_cli
from chess_lab.measure.evaluate import games_benchmark
from chess_lab.models import definitions
from chess_lab.models.store import cards, loader
from tests.conftest import ACCURACY, material_cp, random_positions


def test_games_benchmark_judges_each_quiet_slice(stub_model, tmp_path):
    boards = random_positions(12, seed=3)
    table = pd.DataFrame(
        {
            "fen": [b.fen() for b in boards],
            "cp": [material_cp(b) for b in boards],
            "mate": np.nan,
            "quiet": [i % 3 != 0 for i in range(len(boards))],
        }
    )
    games = tmp_path / "games.parquet"
    table.to_parquet(games, index=False)
    report = games_benchmark(stub_model, games)
    assert report["file"] == "games.parquet"
    assert (report["all"]["n"], report["quiet"]["n"], report["non_quiet"]["n"]) == (12, 8, 4)
    assert report["all"]["excess"] < 1e-3  # the stub scores exactly its label, up to the clip


def test_evaluation_with_a_tag_writes_the_tagged_card(isolated_cards, stub_model, monkeypatch):
    metrics = {
        "accuracy": ACCURACY | {"sample": {"rows": 100, "seed": 1, "agreement_n": 10}},
        "cost": {"us_per_eval": 5.0},
    }
    monkeypatch.setattr(loader, "ARTIFACTS_DIR", isolated_cards / "artifacts")
    monkeypatch.setattr(evaluate_cli, "load_model", lambda *a, **k: stub_model)
    monkeypatch.setattr(evaluate_cli, "sample_benchmark", lambda *a, **k: metrics)
    # sin conjunto de ordenación: el real tiene miles de posiciones
    ranking = isolated_cards / "no-ranking.parquet"
    assert evaluate_cli.main(["--model", "net_pst_w64_both@x", "--ranking", str(ranking)]) == 0
    assert cards.load_card("net_pst_w64_both@x")["accuracy"]["all"]["spearman"] == 0.912
    assert cards.load_card("net_pst_w64_both") == {"name": "net_pst_w64_both"}


def test_evaluation_keeps_the_identity_that_training_wrote(isolated_cards, stub_model, monkeypatch):
    metrics = {"accuracy": ACCURACY | {"sample": {"rows": 1}}, "cost": {"us_per_eval": 5.0}}
    monkeypatch.setattr(loader, "ARTIFACTS_DIR", isolated_cards / "artifacts")
    monkeypatch.setattr(evaluate_cli, "load_model", lambda *a, **k: stub_model)
    monkeypatch.setattr(evaluate_cli, "sample_benchmark", lambda *a, **k: metrics)
    ranking = isolated_cards / "no-ranking.parquet"
    cards.update_card("net_pst_w64_both", architecture="written-by-train", parameters=7)
    assert evaluate_cli.main(["--model", "net_pst_w64_both", "--ranking", str(ranking)]) == 0
    card = cards.load_card("net_pst_w64_both")
    assert card["architecture"] == "written-by-train" and card["parameters"] == 7
    assert card["features"] == stub_model.feature_set  # missing, so filled in


def test_only_registered_models_are_accepted():
    assert "h1" not in definitions.available_models()
    for name in ("h1", "nope"):
        with pytest.raises(SystemExit):
            evaluate_cli.parse_args(["--model", name])
