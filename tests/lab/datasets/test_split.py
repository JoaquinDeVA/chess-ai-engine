"""Tests of the deterministic train/validation/test split."""

import numpy as np
import pandas as pd

from chess_lab.datasets.split import TEST, TRAIN, VALIDATION, assign_split


class TestDatasetSplit:
    def fens(self, n):
        return pd.Series([f"fen-{i} w - -" for i in range(n)])

    def test_split_is_deterministic_and_disjoint(self):
        fens = self.fens(5000)
        first, second = assign_split(fens), assign_split(fens)
        assert (first == second).all()
        assert set(np.unique(first)) == {TRAIN, VALIDATION, TEST}

    def test_fractions_are_respected(self):
        split = assign_split(self.fens(20_000))
        assert 0.08 < (split == TEST).mean() < 0.12
        assert 0.08 < (split == VALIDATION).mean() < 0.12

    def test_a_position_stays_in_its_split_whatever_the_sample(self):
        """The benchmark must not change with row order or ``max_rows``."""
        fens = self.fens(3000)
        full = dict(zip(fens, assign_split(fens), strict=True))
        shuffled = fens.sample(frac=0.5, random_state=3)
        for fen, assigned in zip(shuffled, assign_split(shuffled), strict=True):
            assert full[fen] == assigned

    def test_different_seeds_give_different_splits(self):
        fens = self.fens(2000)
        assert (assign_split(fens, seed=1) != assign_split(fens, seed=2)).any()
