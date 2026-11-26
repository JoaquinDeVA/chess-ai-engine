"""Tests of the training pipeline: targets, dataset split, cards, report and the CLIs."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from chess_lab.datasets import sidecar
from chess_lab.datasets.raw_shards import shard_path, write_shard
from chess_lab.datasets.sidecar import build_sidecar, data_fingerprint
from chess_lab.measure import evaluate as evaluate_cli
from chess_lab.models import definitions
from chess_lab.models.store import cards, loader
from chess_lab.models.target import Representation
from chess_lab.training import train as train_cli
from chess_lab.training.effort import effort_config, fit_config
from chess_lab.training.neural import new_model
from tests.conftest import ACCURACY, material_cp, random_positions


class TestEffortCommandLine:
    """``--effort`` trains a network straight from raw shards and their sidecars."""

    def test_unknown_representation_field_exits_cleanly(self, isolated_cards, tmp_path, capsys):
        code = train_cli.main(
            [
                "--model", "net_pst_w64_both", "--effort", "0",
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

        monkeypatch.setattr(sidecar, "load_sample", recording_load_sample)
        with pytest.raises(StopAfterCallError):
            evaluate_cli.sample_benchmark(stub_model, tmp_path, tmp_path)
        assert calls and calls[0].get("mate_keep", 1.0) == 1.0


class TestMateKeepGuards:
    def test_mate_keep_without_tag_is_refused_and_writes_nothing(
        self, isolated_cards, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(loader, "ARTIFACTS_DIR", tmp_path)
        raw, side = tmp_path / "raw", tmp_path / "side"
        write_mate_shards(raw, side)
        shards = ["--raw", str(raw), "--sidecar", str(side)]

        flag = ["--model", "net_pst_w64_both", "--effort", "0", "--mate-keep", "0.05", *shards]
        assert train_cli.main(flag) == 2
        full = ["--model", "net_pst_w64_both", "--effort", "0", "--mate-keep", "1.0", *shards]
        assert train_cli.main(full) == 2  # 100 % also requires --tag
        representation = [
            "--model", "net_pst_w64_both", "--effort", "0", "--representation",
            '{"mate_keep": 0.05}', *shards,
        ]  # fmt: skip
        assert train_cli.main(representation) == 2
        assert not list(tmp_path.glob("net_pst_w64_both*.npz"))
        assert cards.load_card("net_pst_w64_both") == {"name": "net_pst_w64_both"}


class FakeModel:
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
    monkeypatch.setattr(loader, "ARTIFACTS_DIR", isolated_cards / "artifacts")
    monkeypatch.setattr(train_cli, "count_rows", lambda *a, **k: 1000)
    monkeypatch.setattr(train_cli, "load_sample", lambda *a, **k: None)
    timing = SimpleNamespace(seconds_per_position=1e-5, encode_seconds=0.1, score_seconds=0.1)
    monkeypatch.setattr(train_cli, "calibrate", lambda *a, **k: timing)
    monkeypatch.setattr(train_cli, "estimated_seconds", lambda *a, **k: 1.0)
    monkeypatch.setattr(train_cli, "search", lambda *a, **k: {"l2": 1e-3, "lr": 1e-2})
    monkeypatch.setattr(train_cli, "environment", lambda: {"python": "3"})
    summary = {"train_rows": 900, "val_mae": 0.1, "seconds": 5.0, "config": {}}
    monkeypatch.setattr(train_cli, "fit", lambda *a, **k: (FakeModel(), dict(summary)))
    return isolated_cards


class TestCommandLineWithDoubles:
    """Training and evaluation CLIs with stand-ins: nothing is trained and no metrics are computed."""

    def train(self, *extra):
        argv = [
            "--model",
            "net_pst_w64_both",
            "--effort",
            "40",
            "--raw",
            "r",
            "--sidecar",
            "s",
            *extra,
        ]
        return train_cli.main(argv)

    def test_training_writes_the_card_with_its_training_section(self, fake_training, capsys):
        assert self.train() == 0
        card = cards.load_card("net_pst_w64_both@e40")
        assert card["status"] == "current" and card["parameters"] == 7
        training = card["training"]
        assert training["effort"] == 40 and training["train_samples"] == 900
        assert training["val_mae"] == 0.1 and "effort_config" in training
        assert "representation" in training and "date" in training and "commit" in training
        assert "Effort 40" in capsys.readouterr().out

    def test_init_needs_a_tag(self, fake_training):
        with pytest.raises(SystemExit):
            self.train("--init", "net_pst_w64_both@e15")

    def test_init_starts_the_fit_from_the_source_and_names_it_in_the_card(
        self, fake_training, monkeypatch
    ):
        source = new_model(definitions.get_model_class("net_pst_w64_both"))
        monkeypatch.setattr(train_cli, "load_model", lambda spec: source)
        calls = []

        def recording_fit(*args, **kwargs):
            calls.append(kwargs)
            return FakeModel(), {"train_rows": 900, "seconds": 5.0, "config": {}}

        monkeypatch.setattr(train_cli, "fit", recording_fit)
        assert self.train("--init", "net_pst_w64_both@e15", "--tag", "warm") == 0
        start = calls[0]["initial"]
        assert all((a == b).all() for (a, _), (b, _) in zip(start, source.layers, strict=True))
        assert cards.load_card("net_pst_w64_both@e40-warm")["training"]["init"] == (
            "net_pst_w64_both@e15"
        )

    def test_lr_and_l2_options_replace_the_searched_values_in_the_final_fit(
        self, fake_training, monkeypatch
    ):
        calls = []

        def recording_fit(*args, **kwargs):
            calls.append(args)
            return FakeModel(), {"train_rows": 900, "seconds": 5.0, "config": {}}

        monkeypatch.setattr(train_cli, "fit", recording_fit)
        assert self.train("--lr", "5e-4") == 0
        config = calls[0][6]
        assert config["lr"] == 5e-4 and config["l2"] == 1e-3
        assert self.train("--l2", "2e-5") == 0
        config = calls[1][6]
        assert config["lr"] == 1e-2 and config["l2"] == 2e-5

    def test_the_recipe_is_the_same_at_every_effort_and_reaches_the_card(
        self, fake_training, capsys
    ):
        assert self.train() == 0
        assert "  max_epochs: 100" in capsys.readouterr().out
        training = cards.load_card("net_pst_w64_both@e40")["training"]
        assert training["effort_config"]["patience"] == 3
        assert training["recipe"]["effort"] == 40 and training["recipe"]["init"] is None
        assert training["data"]["shards"] == 0 and "environment" in training
        assert "recipe" not in training["config"]

    def test_trials_option_replaces_the_trials_of_the_effort(self, fake_training, capsys):
        assert self.train("--trials", "12") == 0
        assert "  trials: 12" in capsys.readouterr().out
        assert cards.load_card("net_pst_w64_both@e40")["training"]["recipe"]["trials"] == 12

    def test_a_resumed_fit_counts_the_setup_before_the_cut(self, fake_training, monkeypatch):
        plan = effort_config(40, 1000)
        config = {**fit_config(plan, 1e-3, 1e-2), "setup_seconds": 2.0, "recipe": self.recipe()}
        monkeypatch.setattr(train_cli, "checkpoint_config", lambda path: config)
        assert self.train("--checkpoint", "c.npz") == 0
        assert cards.load_card("net_pst_w64_both@e40")["training"]["seconds"] == 7.0

    def test_a_checkpoint_of_another_recipe_is_refused(self, fake_training, monkeypatch, capsys):
        plan = effort_config(40, 1000)
        config = {**fit_config(plan, 1e-3, 1e-2), "setup_seconds": 2.0, "recipe": self.recipe()}
        monkeypatch.setattr(train_cli, "checkpoint_config", lambda path: config)
        assert self.train("--checkpoint", "c.npz", "--lr", "5e-4") == 2
        assert "recipe.lr" in capsys.readouterr().out
        assert cards.load_card("net_pst_w64_both@e40").get("training") is None

    def recipe(self):
        """The recipe that ``self.train()`` resolves (1000 TRAIN rows available, no shards)."""
        return {
            "model": "net_pst_w64_both", "effort": 40, "positions": 1000, "seed": 42,
            "representation": Representation().to_dict(), "init": None, "lr": None, "l2": None, "trials": 1,
            "data": {"shards": 0, "sha12": data_fingerprint(Path("r"), Path("s"))["sha12"]},
        }  # fmt: skip

    def test_the_artifact_and_card_carry_the_effort_and_a_variant_goes_after_it(
        self, fake_training
    ):
        cards.update_card("net_pst_w64_both@e40", status="current", parameters=99)
        assert self.train("--tag", "x") == 0
        assert (fake_training / "artifacts" / "net_pst_w64_both@e40-x.npz").exists()
        assert cards.load_card("net_pst_w64_both@e40-x")["training"]["effort"] == 40
        assert cards.load_card("net_pst_w64_both@e40") == {
            "name": "net_pst_w64_both@e40", "status": "current", "parameters": 99
        }  # fmt: skip

    def test_retraining_drops_the_measurements_of_the_previous_model(self, fake_training):
        cards.update_card(
            "net_pst_w64_both@e40",
            status="current",
            training={"effort": 15, "old_key": 1},
            accuracy=ACCURACY,
            cost={"us_per_eval": 5.0},
        )
        assert self.train() == 0
        card = cards.load_card("net_pst_w64_both@e40")
        assert not {"accuracy", "cost"} & set(card)
        assert card["training"]["effort"] == 40 and "old_key" not in card["training"]
        assert card["parameters"] == 7


@pytest.mark.parametrize("value", ["0", "-1e-3"])
def test_lr_not_positive_exits_with_code_2(value):
    with pytest.raises(SystemExit) as exit_info:
        train_cli.parse_args(["--model", "net_pst_w64_both", "--effort", "0", "--lr", value])
    assert exit_info.value.code == 2
