"""Tests of the training pipeline: targets, dataset split, cards, report and the CLIs."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chess_engine import models
from chess_engine.comparison import evaluate as evaluate_cli
from chess_engine.comparison import report
from chess_engine.models import base as models_base
from chess_engine.models import cards, registry
from chess_engine.training import train as train_cli
from chess_engine.training.dataset import (
    TEST,
    TRAIN,
    VALIDATION,
    assign_split,
    encode_positions,
    split_dataset,
)
from chess_engine.training.lichess_raw import shard_path, write_shard
from chess_engine.training.representation import Representation, white_score
from chess_engine.training.sidecar import build_sidecar
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


class TestEncoding:
    def test_parallel_encoding_equals_serial_encoding(self):
        extractor = models.features.VectorFeatures()
        fens = pd.Series([" ".join(b.fen().split()[:4]) for b in random_positions(50, seed=3)])
        (serial,) = encode_positions(extractor, fens)
        (parallel,) = encode_positions(extractor, fens, n_jobs=2, chunk_size=10)
        np.testing.assert_array_equal(serial, parallel)


@pytest.fixture
def isolated_cards(tmp_path, monkeypatch):
    monkeypatch.setattr(cards, "CARDS_DIR", tmp_path / "cards")
    monkeypatch.setattr(report, "HISTORY_FILE", tmp_path / "RESULTS.md")
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

    def test_clear_measurements_keeps_only_the_identity(self, isolated_cards):
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
            tradeoff=TRADEOFF,
            tournament={"score": 50.0},
        )
        cards.clear_measurements("m")
        assert cards.load_card("m") == {
            "name": "m", "status": "current", "architecture": "a", "features": "f",
            "parameters": 3, "size_kb": 1.5,
        }  # fmt: skip
        assert json.loads(cards.card_path("m").read_text()) == cards.load_card("m")

    def test_clear_measurements_of_a_missing_card_does_nothing(self, isolated_cards):
        cards.clear_measurements("ghost")
        assert not cards.card_path("ghost").exists()

    def test_missing_card_is_an_empty_skeleton(self, isolated_cards):
        assert cards.load_card("ghost") == {"name": "ghost"}

    def test_report_renders_new_columns(self, isolated_cards):
        representation = Representation().to_dict()
        cards.update_card(
            "linear_pst",
            status="current",
            architecture="PST",
            parameters=1234,
            training={"effort": 40, "representation": representation},
            accuracy=ACCURACY,
            cost={"us_per_eval": 12.5},
            tradeoff=TRADEOFF,
            tournament={"600+5": {"games": 400, "score": 56.2, "elo": 43.0, "elo_interval": 24.1}},
        )
        text = report.render_tables()
        header = (
            "| Model | Kind | Params | Representation | Effort | Spearman | MAE | "
            "Best move, all / non-quiet | µs per eval | Depth 1 s | Depth 3 s | Depth 10 s | "
            "Search agreement, 10 s | 10 min + 5 s, score % | 10 min + 5 s, Elo vs anchor |"
        )
        row = (
            f"| `linear_pst` | plays | 1,234 | {Representation().name} | 40 | 0.912 | 0.083 | "
            "34.5% / 28.0% | 12.5 | 1.0 | 3.2 | 6.0 | 40.0% | 56.2% | +43 ± 24 |"
        )
        assert header in text and row in text

    def test_report_orders_controls_and_dashes_missing(self, isolated_cards):
        def entry(score, elo):
            return {"score": score, "elo": elo, "elo_interval": 10.0}

        cards.update_card(
            "a",
            status="current",
            tournament={
                "depth3": entry(70.0, 90.0),
                "60+1": entry(40.0, -35.0),
                "5+0": entry(1.0, 2.0),
            },
        )
        cards.update_card("b", status="current", tournament={"600+5": entry(55.0, 20.0)})
        cards.update_card(
            "c",
            status="current",
            tournament={
                "base": 60.0,
                "increment": 1.0,
                "score": 30.0,
                "elo": -60.0,
                "elo_interval": 8.0,
            },
        )
        lines = report.render_tables().splitlines()
        titles = [t.strip() for t in lines[0].strip("|").split("|")][13:]
        assert titles == [
            "10 min + 5 s, score %", "10 min + 5 s, Elo vs anchor",
            "1 min + 1 s, score %", "1 min + 1 s, Elo vs anchor",
            "Depth 3, score %", "Depth 3, Elo vs anchor",
            "5+0, score %", "5+0, Elo vs anchor",
        ]  # fmt: skip
        cells = {
            line.split("|")[1].strip(): [c.strip() for c in line.strip("|").split("|")][13:]
            for line in lines[2:]
        }
        assert cells["`a`"] == [
            "—",
            "—",
            "40.0%",
            "-35 ± 10",
            "70.0%",
            "+90 ± 10",
            "1.0%",
            "+2 ± 10",
        ]
        assert cells["`b`"] == ["55.0%", "+20 ± 10", *["—"] * 6]
        assert cells["`c`"] == ["—", "—", "30.0%", "-60 ± 8", *["—"] * 4]  # old flat layout

    def test_accuracy_record_models_marked(self, isolated_cards):
        cards.update_card(
            "mlp_features",
            status="current",
            parameters=1505,
            accuracy=ACCURACY,
            cost={"us_per_eval": 12.5},
        )
        text = report.render_tables()
        assert (
            "| `mlp_features` | accuracy record | 1,505 | — | — | 0.912 | 0.083 | 34.5% / 28.0% "
            "| 12.5 | — | — | — | — |"
        ) in text

    def test_models_without_sections_show_dashes(self, isolated_cards):
        cards.update_card("fresh", status="current", architecture="A")
        assert f"| `fresh` | plays |{' — |' * 11}" in report.render_tables()

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


class TestAccuracyRecords:
    def test_csv_family_models_are_not_trained(self, capsys):
        for name in ("mlp_features", "cnn_resnet"):
            assert train_cli.main(["--model", name, "--effort", "0"]) == 2
            assert f"{name} is an accuracy record" in capsys.readouterr().out


class TestEffortCommandLine:
    """``--effort`` trains a linear model straight from raw shards and their sidecars."""

    def test_unknown_representation_field_exits_cleanly(self, isolated_cards, tmp_path, capsys):
        code = train_cli.main(
            [
                "--model", "linear_pst", "--effort", "0",
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

        flag = ["--model", "linear_pst", "--effort", "0", "--mate-keep", "0.05", *shards]
        assert train_cli.main(flag) == 2
        full = ["--model", "linear_pst", "--effort", "0", "--mate-keep", "1.0", *shards]
        assert train_cli.main(full) == 2  # 100 % also requires --tag
        representation = [
            "--model", "linear_pst", "--effort", "0", "--representation",
            '{"mate_keep": 0.05}', *shards,
        ]  # fmt: skip
        assert train_cli.main(representation) == 2
        assert not list(tmp_path.glob("linear_pst*.npz"))
        assert cards.load_card("linear_pst") == {"name": "linear_pst"}


class FakeLinearModel:
    """Stand-in for the trained model: saves a minimal file and uses neither data nor TensorFlow."""

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")
        return path

    def count_parameters(self):
        return 7


@pytest.fixture
def fake_training(isolated_cards, monkeypatch):
    """Replaces the data reading, calibration, search and fit of ``train_by_effort``."""
    monkeypatch.setattr(registry, "ARTIFACTS_DIR", isolated_cards / "artifacts")
    monkeypatch.setattr(models_base, "ARTIFACTS_DIR", isolated_cards / "artifacts")
    monkeypatch.setattr(train_cli, "available_rows", lambda *a, **k: 1000)
    monkeypatch.setattr(train_cli, "load_sample", lambda *a, **k: None)
    timing = SimpleNamespace(seconds_per_position=1e-5, encode_seconds=0.1, score_seconds=0.1)
    monkeypatch.setattr(train_cli, "calibrate", lambda *a, **k: timing)
    monkeypatch.setattr(train_cli, "estimated_seconds", lambda *a, **k: 1.0)
    monkeypatch.setattr(train_cli, "search", lambda *a, **k: {"l2": 1e-3, "lr": 1e-2})
    summary = {"train_rows": 900, "val_mae": 0.1}
    monkeypatch.setattr(train_cli, "fit", lambda *a, **k: (FakeLinearModel(), dict(summary)))
    return isolated_cards


class TestCommandLineWithDoubles:
    """Training and evaluation CLIs with stand-ins: nothing is trained and no metrics are computed."""

    def train(self, *extra):
        argv = ["--model", "linear_pst", "--effort", "40", "--raw", "r", "--sidecar", "s", *extra]
        return train_cli.main(argv)

    def test_training_writes_the_card_with_its_training_section(self, fake_training, capsys):
        assert self.train() == 0
        card = cards.load_card("linear_pst")
        assert card["status"] == "current" and card["parameters"] == 7
        training = card["training"]
        assert training["effort"] == 40 and training["train_samples"] == 900
        assert training["val_mae"] == 0.1 and "effort_config" in training
        assert "representation" in training and "date" in training and "commit" in training
        assert "Effort 40" in capsys.readouterr().out

    def test_tag_writes_artifact_and_card_without_touching_the_untagged_one(self, fake_training):
        cards.update_card("linear_pst", status="current", parameters=99)
        assert self.train("--tag", "x") == 0
        assert (fake_training / "artifacts" / "linear_pst@x.npz").exists()
        assert cards.load_card("linear_pst@x")["training"]["effort"] == 40
        assert cards.load_card("linear_pst") == {
            "name": "linear_pst", "status": "current", "parameters": 99
        }  # fmt: skip

    def test_evaluation_with_a_tag_writes_the_tagged_card(
        self, isolated_cards, stub_model, monkeypatch
    ):
        metrics = {
            "accuracy": ACCURACY | {"sample": {"rows": 100, "seed": 1, "agreement_n": 10}},
            "cost": {"us_per_eval": 5.0},
        }
        monkeypatch.setattr(registry, "ARTIFACTS_DIR", isolated_cards / "artifacts")
        monkeypatch.setattr(evaluate_cli, "load_model", lambda *a, **k: stub_model)
        monkeypatch.setattr(evaluate_cli, "sample_benchmark", lambda *a, **k: metrics)
        assert evaluate_cli.main(["--model", "linear_pst@x"]) == 0
        assert cards.load_card("linear_pst@x")["accuracy"]["all"]["spearman"] == 0.912
        assert cards.load_card("linear_pst") == {"name": "linear_pst"}

    def test_retraining_drops_the_measurements_of_the_previous_model(self, fake_training):
        cards.update_card(
            "linear_pst",
            status="current",
            training={"effort": 15, "old_key": 1},
            accuracy=ACCURACY,
            cost={"us_per_eval": 5.0},
            tradeoff=TRADEOFF,
            tournament={"games": 10, "score": 50.0},
        )
        assert self.train() == 0
        card = cards.load_card("linear_pst")
        assert not {"accuracy", "cost", "tradeoff", "tournament"} & set(card)
        assert card["training"]["effort"] == 40 and "old_key" not in card["training"]
        assert card["parameters"] == 7


class TestHeuristicEvaluation:
    def test_h1_is_accepted_and_unknown_names_are_not(self):
        assert evaluate_cli.parse_args(["--model", "h1"]).model == "h1"
        with pytest.raises(SystemExit):
            evaluate_cli.parse_args(["--model", "nope"])
        assert "h1" not in registry.available_models()

    def test_adapter_scores_are_recovered_by_the_judge_scale(self):
        model = evaluate_cli.HeuristicModel("h1")
        boards = random_positions(5, seed=3, max_plies=12)
        recovered = white_score(model.predict_cp_batch(boards), np.nan, Representation())
        expected = [model.expected_score(board) for board in boards]
        assert recovered == pytest.approx(expected)
        assert all(0.01 <= score <= 0.99 for score in expected)

    def test_main_writes_a_reference_card_that_the_report_lists(self, isolated_cards, monkeypatch):
        boards = random_positions(30, seed=5, max_plies=20)
        sample = pd.DataFrame(
            {
                "fen": [b.fen() for b in boards],
                "cp": [float(material_cp(b)) for b in boards],
                "mate": [np.nan] * len(boards),
                "best_move": [next(iter(b.legal_moves)).uci() for b in boards],
                "quiet": [i % 2 == 0 for i in range(len(boards))],
            }
        )
        monkeypatch.setattr(evaluate_cli, "load_sample", lambda *a, **k: sample)
        assert evaluate_cli.main(["--model", "h1", "--agreement-n", "5"]) == 0
        card = cards.load_card("h1")
        assert card["status"] == "reference" and card["features"] == "h1"
        assert card["architecture"] == "Hand-crafted heuristic (weights fitted to Stockfish)"
        assert card["parameters"] == 6 and "training" not in card and "size_kb" not in card
        assert card["accuracy"]["all"]["n"] == 30 and card["cost"]["us_per_eval"] > 0
        assert "| `h1` | plays | 6 |" in report.render_tables()


@pytest.mark.parametrize("value", ["1.5", "1", "0", "-0.5"])
def test_lr_decay_outside_zero_one_exits_with_code_2(value):
    with pytest.raises(SystemExit) as exit_info:
        train_cli.parse_args(["--model", "linear_pst", "--effort", "0", "--lr-decay", value])
    assert exit_info.value.code == 2
