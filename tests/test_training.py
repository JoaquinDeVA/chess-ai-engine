"""Tests of the training pipeline: targets, dataset split, metrics, cards, report and the CLIs."""

import json

import numpy as np
import pandas as pd
import pytest

from chess_engine import models
from chess_engine.models import registry
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
from chess_engine.training.lichess_raw import shard_path, write_shard
from chess_engine.training.metrics import compute_metrics, outcome_class
from chess_engine.training.representation import Representation
from chess_engine.training.sidecar import build_sidecar
from chess_engine.training.targets import cp_to_target, mate_to_target, row_targets
from tests.conftest import material_cp, random_positions
from tests.test_effort import write_fixture_shards


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


ACCURACY = {
    "all": {"spearman": 0.912, "mae": 0.083, "best_move_agreement": 0.345, "n": 100},
    "quiet": {"spearman": 0.9, "mae": 0.07, "best_move_agreement": 0.4, "n": 60},
    "non_quiet": {"spearman": 0.8, "mae": 0.1, "best_move_agreement": 0.28, "n": 40},
}
TRADEOFF = {
    "1": {"mean_depth": 1.0, "search_agreement": 0.2},
    "3": {"mean_depth": 3.2, "search_agreement": 0.3},
    "10": {"mean_depth": 6.0, "search_agreement": 0.4},
}


class TestCardsAndReport:
    def test_update_merges_dictionaries_and_replaces_scalars(self, isolated_cards):
        cards.update_card("m", architecture="a", training={"epochs": 3, "dataset": "x"})
        card = cards.update_card("m", architecture="b", training={"epochs": 4})
        assert card["architecture"] == "b"
        assert card["training"] == {"epochs": 4, "dataset": "x"}
        assert json.loads(cards.card_path("m").read_text())["training"]["epochs"] == 4

    def test_missing_card_is_an_empty_skeleton(self, isolated_cards):
        assert cards.load_card("ghost") == {"name": "ghost"}

    def test_report_renders_new_columns(self, isolated_cards):
        representation = Representation().to_dict()
        cards.update_card(
            "pst_tapered",
            status="current",
            architecture="PST",
            parameters=1234,
            training={"effort": 40, "representation": representation},
            accuracy=ACCURACY,
            cost={"us_per_eval": 12.5},
            tradeoff=TRADEOFF,
            tournament={"games": 400, "score": 56.2, "elo": 43.0, "elo_interval": 24.1},
        )
        text = report.render_tables()
        header = (
            "| Model | Kind | Params | Representation | Effort | Spearman | MAE | "
            "Best move, all / non-quiet | µs per eval | Depth 1 s | Depth 3 s | Depth 10 s | "
            "Search agreement, 10 s | Score | Elo ± interval |"
        )
        row = (
            f"| `pst_tapered` | plays | 1,234 | {Representation().name} | 40 | 0.912 | 0.083 | "
            "34.5% / 28.0% | 12.5 | 1.0 | 3.2 | 6.0 | 40.0% | 56.2% | +43 ± 24 |"
        )
        assert header in text and row in text

    def test_accuracy_record_models_marked(self, isolated_cards):
        cards.update_card(
            "ridge_features",
            status="current",
            parameters=30,
            accuracy=ACCURACY,
            cost={"us_per_eval": 12.5},
        )
        text = report.render_tables()
        assert (
            "| `ridge_features` | accuracy record | 30 | — | — | 0.912 | 0.083 | 34.5% / 28.0% | 12.5 "
            "| — | — | — | — | — | — |"
        ) in text

    def test_models_without_sections_show_dashes(self, isolated_cards):
        cards.update_card("fresh", status="current", architecture="A")
        assert f"| `fresh` | plays |{' — |' * 13}" in report.render_tables()

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
        raw, side = tmp_path / "raw", tmp_path / "side"
        write_fixture_shards(raw, side, 900, seed=4)
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

    def test_effort_40_runs_one_trial_and_records_it(self, isolated_cards, tmp_path):
        raw, side = tmp_path / "raw", tmp_path / "side"
        write_fixture_shards(raw, side, 900, seed=4)
        target = tmp_path / "pst_tapered.npz"

        assert (
            train_cli.main(
                [
                    "--model", "pst_tapered", "--effort", "40",
                    "--raw", str(raw), "--sidecar", str(side), "--output", str(target),
                ]
            )
            == 0
        )  # fmt: skip
        training = cards.load_card("pst_tapered")["training"]
        assert training["effort"] == 40
        assert training["effort_config"]["trials"] == 1
        assert training["effort_config"]["max_epochs"] == 5

    def test_evaluate_writes_accuracy_and_cost_from_the_test_sample(self, isolated_cards, tmp_path):
        raw, side = tmp_path / "raw", tmp_path / "side"
        write_fixture_shards(raw, side, 900, seed=4)
        target = tmp_path / "pst_tapered.npz"
        shards = ["--raw", str(raw), "--sidecar", str(side)]
        assert (
            train_cli.main(
                ["--model", "pst_tapered", "--effort", "0", *shards, "--output", str(target)]
            )
            == 0
        )

        assert (
            evaluate_cli.main(
                ["--model", "pst_tapered", "--path", str(target), *shards, "--agreement-n", "50"]
            )
            == 0
        )
        card = cards.load_card("pst_tapered")
        accuracy = card["accuracy"]
        rows = accuracy["sample"]["rows"]
        assert rows > 0 and accuracy["sample"] == {"rows": rows, "seed": 42, "agreement_n": 50}
        assert set(accuracy) >= {"all", "quiet", "non_quiet", "sample"}
        assert accuracy["all"]["n"] == rows and 0 <= accuracy["all"]["mae"] <= 1
        assert card["cost"]["us_per_eval"] > 0
        assert "benchmark" not in card

    def test_unknown_representation_field_exits_cleanly(self, isolated_cards, tmp_path, capsys):
        code = train_cli.main(
            [
                "--model", "pst_tapered", "--effort", "0",
                "--raw", str(tmp_path), "--sidecar", str(tmp_path),
                "--representation", '{"bogus": 1}',
            ]
        )  # fmt: skip
        assert code == 2
        assert "bogus" in capsys.readouterr().out


def write_mate_shards(raw, side, count=900, seed=4):
    """Like :func:`write_fixture_shards`, but every fifth position is a mate in 2."""
    raw.mkdir(parents=True, exist_ok=True)
    boards = random_positions(count, seed=seed)
    records = []
    for index, board in enumerate(boards):
        fen = " ".join(board.fen().split()[:4])
        record = {"fen": fen, "cp": material_cp(board), "mate": None, "depth": 20, "knodes": 1,
                  "best_move": None, "n_evals": 1, "n_pvs": 1}  # fmt: skip
        if index % 5 == 0:
            record.update(cp=None, mate=2)
        records.append(record)
    write_shard(shard_path(raw, 0), records, {"perspective": "white"})
    build_sidecar(shard_path(raw, 0), side)


class TestMateKeepAndTags:
    def test_train_cli_mate_keep_and_tag_write_tagged_artifact_and_card(
        self, isolated_cards, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(registry, "ARTIFACTS_DIR", tmp_path)
        raw, side = tmp_path / "raw", tmp_path / "side"
        write_mate_shards(raw, side)

        code = train_cli.main(
            [
                "--model", "pst_tapered", "--effort", "0", "--mate-keep", "0.05", "--tag", "mates5",
                "--raw", str(raw), "--sidecar", str(side),
            ]
        )  # fmt: skip
        assert code == 0
        assert (tmp_path / "pst_tapered@mates5.npz").exists()
        assert not (tmp_path / "pst_tapered.npz").exists()
        card = cards.load_card("pst_tapered@mates5")
        assert card["name"] == "pst_tapered@mates5"
        assert card["training"]["representation"]["mate_keep"] == 0.05
        assert cards.load_card("pst_tapered") == {"name": "pst_tapered"}

    def test_judge_sample_ignores_mate_keep(self, stub_model, tmp_path, monkeypatch):
        """The test sample of the evaluation is always read with every row (no ``mate_keep``)."""

        class StopAfterCallError(Exception):
            pass

        calls = []

        def recording_load_sample(*args, **kwargs):
            calls.append(kwargs)
            raise StopAfterCallError

        monkeypatch.setattr(evaluate_cli, "load_sample", recording_load_sample)
        with pytest.raises(StopAfterCallError):
            evaluate_cli.sample_benchmark(stub_model, tmp_path, tmp_path)
        assert calls and calls[0].get("mate_keep", 1.0) == 1.0


class TestMateKeepGuards:
    def test_mate_keep_without_tag_is_refused_and_writes_nothing(
        self, isolated_cards, tmp_path, monkeypatch
    ):
        from chess_engine.models import base as models_base

        monkeypatch.setattr(models_base, "ARTIFACTS_DIR", tmp_path)
        monkeypatch.setattr(registry, "ARTIFACTS_DIR", tmp_path)
        raw, side = tmp_path / "raw", tmp_path / "side"
        write_mate_shards(raw, side)
        shards = ["--raw", str(raw), "--sidecar", str(side)]

        flag = ["--model", "pst_tapered", "--effort", "0", "--mate-keep", "0.05", *shards]
        assert train_cli.main(flag) == 2
        representation = [
            "--model", "pst_tapered", "--effort", "0", "--representation",
            '{"mate_keep": 0.05}', *shards,
        ]  # fmt: skip
        assert train_cli.main(representation) == 2
        assert not list(tmp_path.glob("pst_tapered*.npz"))
        assert cards.load_card("pst_tapered") == {"name": "pst_tapered"}

    def test_evaluate_accepts_a_tagged_spec_and_writes_its_card(
        self, isolated_cards, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(registry, "ARTIFACTS_DIR", tmp_path)
        raw, side = tmp_path / "raw", tmp_path / "side"
        write_mate_shards(raw, side)
        shards = ["--raw", str(raw), "--sidecar", str(side)]
        tagged = ["--model", "pst_tapered", "--effort", "0", "--mate-keep", "0.05",
                  "--tag", "mates5", *shards]  # fmt: skip
        assert train_cli.main(tagged) == 0

        assert (
            evaluate_cli.main(
                ["--model", "pst_tapered@mates5", "--path",
                 str(tmp_path / "pst_tapered@mates5.npz"), *shards, "--agreement-n", "50"]
            )
            == 0
        )  # fmt: skip
        card = cards.load_card("pst_tapered@mates5")
        assert card["accuracy"]["sample"]["rows"] > 0 and card["cost"]["us_per_eval"] > 0
        assert "accuracy" not in cards.load_card("pst_tapered")
        with pytest.raises(SystemExit):
            evaluate_cli.parse_args(["--model", "no_such_model@x"])
