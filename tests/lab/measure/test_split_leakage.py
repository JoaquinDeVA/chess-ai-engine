"""Tests of the accuracy-by-neighbour measurement."""

import numpy as np
import pandas as pd

from chess_lab.measure import split_leakage
from tests.conftest import material_cp, random_positions


def test_each_model_is_judged_on_the_positions_with_and_without_a_neighbour(
    stub_model, monkeypatch
):
    monkeypatch.setattr(split_leakage, "load_evaluator", lambda name: stub_model)
    boards = random_positions(12, seed=5)
    sample = pd.DataFrame(
        {
            "fen": [b.fen() for b in boards],
            "cp": [material_cp(b) for b in boards],
            "mate": np.nan,
            "best_move": [next(iter(b.legal_moves)).uci() if b.legal_moves else "" for b in boards],
            "quiet": True,
        }
    )
    flags = np.array([i % 4 == 0 for i in range(12)])
    report = split_leakage.stratified_accuracy(["net_x"], sample, flags)
    groups = report["net_x"]
    assert (groups["with neighbour"]["rows"], groups["without neighbour"]["rows"]) == (3, 9)
    lines = split_leakage.report_lines(report, len(sample), flags.mean())
    assert lines[0].startswith("## Accuracy") and "`net_x`" in lines[-1]
