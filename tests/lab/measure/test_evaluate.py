"""Tests of the evaluation CLI on the game-position set."""

import numpy as np
import pandas as pd

from chess_lab.measure.evaluate import games_benchmark
from tests.conftest import material_cp, random_positions


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
