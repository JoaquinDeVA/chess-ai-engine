"""Tests of the training pipeline: targets, dataset split, metrics, cards, report and the CLIs."""

import json

import numpy as np
import pandas as pd
import pytest

from chess_engine import models
from chess_engine.training import cards, report
from chess_engine.training import evaluate as evaluate_cli
from chess_engine.training import train as train_cli
from chess_engine.training.dataset import (
    TEST,
    TRAIN,
    VALIDATION,
    assign_split,
    encode_positions,
    load_evaluations,
    split_dataset,
)
from chess_engine.training.metrics import compute_metrics, outcome_class
from chess_engine.training.representation import Representation
from chess_engine.training.targets import cp_to_target, mate_to_target, row_targets
from tests.conftest import material_cp, random_positions


class TestTargets:
    def test_centipawns_are_normalized_and_clipped(self):
        np.testing.assert_allclose(
            cp_to_target([0, 633.5, -1267, 5000, -20000]), [0, 0.5, -1, 1, -1]
        )

    def test_sign_convention_is_white_point_of_view(self):
        """The course notebooks divided by a *negative* constant, inverting every cp target."""
        assert cp_to_target(300) > 0 > cp_to_target(-300)

    def test_mates_keep_their_sign_and_far_mates_are_dropped(self):
        result = mate_to_target([1, -2, 3, -4])
        assert result[:2].tolist() == [1.0, -1.0] and np.isnan(result[2:]).all()

    def test_mate_and_cp_targets_agree_in_sign(self):
        targets = row_targets(
            [False, True, False, True], [250, np.nan, -250, np.nan], [np.nan, 1, np.nan, -1]
        )
        assert np.sign(targets).tolist() == [1, 1, -1, -1]

    def test_row_targets_mixes_rows_and_marks_dropped_ones_with_nan(self):
        targets = row_targets([False, True, True], [633.5, np.nan, np.nan], [np.nan, 2, 9])
        assert targets[0] == 0.5 and targets[1] == 1.0 and np.isnan(targets[2])

    def test_a_custom_mate_horizon_is_respected(self):
        assert row_targets([True], [np.nan], [4], max_mate_distance=5)[0] == 1.0
        assert np.isnan(row_targets([True], [np.nan], [4], max_mate_distance=3)[0])


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

    def test_split_dataset_partitions_the_frame(self):
        frame = pd.DataFrame({"fen": self.fens(3000), "y": 0.0})
        train, validation, test = split_dataset(frame)
        assert len(train) + len(validation) + len(test) == len(frame)
        assert not (set(train.fen) & set(test.fen))


def write_csv(path, boards, mates=()):
    rows = [
        {
            "fen": " ".join(b.fen().split()[:4]),
            "cp": material_cp(b),
            "is_mate": False,
            "mate_distance": np.nan,
        }
        for b in boards
    ]
    rows += [
        {"fen": fen, "cp": np.nan, "is_mate": True, "mate_distance": distance}
        for fen, distance in mates
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestLoading:
    def test_loads_targets_and_drops_distant_mates_and_duplicates(self, tmp_path):
        boards = random_positions(30, seed=1)
        path = write_csv(
            tmp_path / "evals.csv",
            boards + boards[:3],
            mates=[("6k1/5ppp/8/8/8/8/8/R3K3 w - -", 1), ("6k1/5ppp/8/8/8/8/8/R3K3 b - -", 7)],
        )
        frame = load_evaluations(path)
        assert list(frame.columns) == ["fen", "y"]
        assert frame.fen.is_unique
        assert (frame.y.abs() <= 1).all()
        assert frame.loc[frame.fen == "6k1/5ppp/8/8/8/8/8/R3K3 w - -", "y"].item() == 1.0
        assert not (frame.fen == "6k1/5ppp/8/8/8/8/8/R3K3 b - -").any()

    def test_max_rows_reads_only_the_head(self, tmp_path):
        path = write_csv(tmp_path / "evals.csv", random_positions(40, seed=2))
        assert len(load_evaluations(path, max_rows=10)) <= 10

    def test_missing_columns_are_reported(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("fen,score\nx,1\n")
        with pytest.raises(ValueError, match="missing columns"):
            load_evaluations(path)

    def test_parallel_encoding_equals_serial_encoding(self):
        extractor = models.features.VectorFeatures()
        fens = pd.Series([" ".join(b.fen().split()[:4]) for b in random_positions(50, seed=3)])
        (serial,) = encode_positions(extractor, fens)
        (parallel,) = encode_positions(extractor, fens, n_jobs=2, chunk_size=10)
        np.testing.assert_array_equal(serial, parallel)


class TestMetrics:
    def test_outcome_classes_use_a_dead_zone_of_50_cp(self):
        assert outcome_class([-51, -50, 0, 50, 51]).tolist() == [-1, 0, 0, 0, 1]

    def test_perfect_predictions(self):
        truth = np.array([-800.0, -20.0, 0.0, 120.0, 900.0])
        metrics = compute_metrics(truth, truth)
        assert metrics["mae_cp"] == 0 and metrics["accuracy"] == 1 and metrics["within_50"] == 1

    def test_known_errors(self):
        metrics = compute_metrics([0, 100, 300], [0, 0, 0])
        assert metrics["mae_cp"] == pytest.approx(400 / 3)
        assert metrics["rmse_cp"] == pytest.approx(np.sqrt(100_000 / 3))
        assert metrics["within_50"] == pytest.approx(1 / 3)
        assert metrics["within_100"] == pytest.approx(2 / 3)
        assert metrics["accuracy"] == pytest.approx(1 / 3)

    def test_a_small_disagreement_inside_the_equal_zone_is_not_a_miss(self):
        assert compute_metrics([30.0], [-40.0])["accuracy"] == 1.0

    def test_opposite_signs_are_a_miss(self):
        assert compute_metrics([400.0], [-400.0])["accuracy"] == 0.0

    @pytest.mark.parametrize("a,b", [([], []), ([1.0], [1.0, 2.0])])
    def test_invalid_inputs_are_rejected(self, a, b):
        with pytest.raises(ValueError):
            compute_metrics(a, b)


@pytest.fixture
def isolated_cards(tmp_path, monkeypatch):
    monkeypatch.setattr(cards, "CARDS_DIR", tmp_path / "cards")
    monkeypatch.setattr(report, "HISTORY_FILE", tmp_path / "MODELS_HISTORY.md")
    return tmp_path


class TestCardsAndReport:
    def test_update_merges_dictionaries_and_replaces_scalars(self, isolated_cards):
        cards.update_card("m", architecture="a", training={"epochs": 3, "dataset": "x"})
        card = cards.update_card("m", architecture="b", training={"epochs": 4})
        assert card["architecture"] == "b"
        assert card["training"] == {"epochs": 4, "dataset": "x"}
        assert json.loads(cards.card_path("m").read_text())["training"]["epochs"] == 4

    def test_missing_card_is_an_empty_skeleton(self, isolated_cards):
        assert cards.load_card("ghost") == {"name": "ghost"}

    def test_table_renders_a_benchmarked_model(self, isolated_cards):
        cards.update_card(
            "good",
            status="current",
            architecture="Ridge",
            parameters=30,
            size_kb=2.0,
            training={"train_samples": 1000, "dataset": "d.csv", "epochs": 1},
            benchmark={
                "ms_per_eval": 1.5,
                "mae_cp": 120.0,
                "rmse_cp": 200.0,
                "accuracy": 0.7,
                "within_50": 0.3,
                "within_100": 0.5,
            },
            strength={"depth": 2, "results": {"H2": "8-1-1"}},
        )
        text = report.render_tables()
        assert (
            "| `good` | current | Ridge | 30 | 2 KB | 1,000 positions, d.csv, 1 epochs | 1.50 ms | 120.0 |"
            in text
        )
        assert "70.0%" in text and "H2 8-1-1 (depth 2)" in text

    def test_models_without_a_benchmark_show_dashes(self, isolated_cards):
        cards.update_card("fresh", status="current", architecture="A")
        assert "| — | — | — | — | — | — |" in report.render_tables()

    def test_history_file_is_updated_between_markers_only(self, isolated_cards):
        path = report.HISTORY_FILE
        path.write_text(f"intro\n{report.BEGIN_MARKER}\nSTALE\n{report.END_MARKER}\noutro\n")
        cards.update_card("m", status="current", architecture="A")
        assert report.update_history_file()
        text = path.read_text()
        assert (
            text.startswith("intro\n")
            and text.endswith("\noutro\n")
            and "STALE" not in text
            and "`m`" in text
        )

    def test_history_file_without_markers_is_left_alone(self, isolated_cards):
        report.HISTORY_FILE.write_text("no markers here")
        assert not report.update_history_file()
        assert report.HISTORY_FILE.read_text() == "no markers here"
        assert not report.update_history_file(isolated_cards / "does-not-exist.md")


class TestCommandLines:
    """End-to-end: CSV -> train -> save -> evaluate -> card, in a temporary directory."""

    def test_train_then_evaluate_ridge_features(self, isolated_cards, tmp_path, monkeypatch):
        boards = random_positions(1200, seed=9)
        csv = write_csv(tmp_path / "evals.csv", boards)
        target = tmp_path / "ridge_features.pkl"

        assert (
            train_cli.main(
                [
                    "--model",
                    "ridge_features",
                    "--data",
                    str(csv),
                    "--output",
                    str(target),
                    "--jobs",
                    "1",
                ]
            )
            == 0
        )
        assert target.exists()
        card = cards.load_card("ridge_features")
        assert (
            card["status"] == "current"
            and card["parameters"] == 30
            and card["training"]["train_samples"] > 500
        )

        assert (
            evaluate_cli.main(
                [
                    "--model",
                    "ridge_features",
                    "--data",
                    str(csv),
                    "--path",
                    str(target),
                    "--latency-samples",
                    "20",
                    "--jobs",
                    "1",
                ]
            )
            == 0
        )
        bench = cards.load_card("ridge_features")["benchmark"]
        assert bench["n_test"] > 50 and bench["ms_per_eval"] > 0
        assert bench["accuracy"] > 0.6, (
            "material regression must beat chance on material-labelled data"
        )

    def test_frozen_models_refuse_to_train(self, tmp_path, capsys):
        csv = write_csv(tmp_path / "evals.csv", random_positions(10, seed=1))
        assert train_cli.main(["--model", "cnn_light", "--data", str(csv)]) == 2
        assert "cannot be retrained" in capsys.readouterr().out

    def test_config_files_are_read(self, tmp_path):
        config = tmp_path / "c.toml"
        config.write_text("alpha = 3.5\n")
        assert train_cli.load_config("ridge_features", config) == {"alpha": 3.5}
        assert train_cli.load_config("ridge_features", tmp_path / "missing.toml") == {}
        assert train_cli.load_config("ridge_features")["alpha"] == 10.0


class TestEffortCommandLine:
    """``--effort`` trains a linear model straight from raw shards and their sidecars."""

    def test_train_cli_trains_pst_on_fixture_shards(self, isolated_cards, tmp_path):
        from chess_engine.training.lichess_raw import shard_path, write_shard
        from chess_engine.training.sidecar import build_sidecar

        raw, side = tmp_path / "raw", tmp_path / "side"
        raw.mkdir()
        boards = random_positions(900, seed=4)
        records = [
            {"fen": " ".join(b.fen().split()[:4]), "cp": material_cp(b), "mate": None,
             "depth": 20, "knodes": 1, "best_move": None, "n_evals": 1, "n_pvs": 1}  # fmt: skip
            for b in boards
        ]  # fmt: skip
        write_shard(shard_path(raw, 0), records, {"perspective": "white"})
        build_sidecar(shard_path(raw, 0), side)
        target = tmp_path / "pst_tapered.npz"

        assert (
            train_cli.main(
                [
                    "--model", "pst_tapered", "--effort", "0",
                    "--raw", str(raw), "--sidecar", str(side), "--output", str(target),
                ]
            )
            == 0
        )  # fmt: skip
        assert target.exists()
        training = cards.load_card("pst_tapered")["training"]
        assert training["effort"] == 0
        assert training["representation"] == Representation().to_dict()
        assert training["train_samples"] > 0 and training["seconds"] > 0
        assert training["effort_config"]["trials"] == 0
