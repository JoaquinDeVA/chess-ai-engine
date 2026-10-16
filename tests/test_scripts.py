"""Lichess conversion, Stockfish labelling and plotting."""

import importlib.util
import io
import json
from pathlib import Path

import chess
import pytest

from chess_engine.models import cards, registry
from chess_engine.models.registry import artifact_path
from chess_engine.training import effort
from chess_engine.training.lichess import (
    SIDE_TO_MOVE,
    WHITE,
    convert,
    detect_perspective,
    parse_record,
    to_row,
    write_csv,
)
from chess_engine.training.lichess_raw import shard_path, write_shard
from chess_engine.training.sidecar import build_sidecar
from chess_engine.training.stockfish import find_stockfish, label_positions
from tests.conftest import material_cp, random_positions

ROOT = Path(__file__).resolve().parent.parent


def database_line(board: chess.Board, perspective: str, mate: int | None = None) -> str:
    """A fake database line whose evaluation follows material, from the given perspective."""
    cp = int(material_cp(board))
    if perspective == SIDE_TO_MOVE and board.turn == chess.BLACK:
        cp = -cp
    pv = {"mate": mate} if mate is not None else {"cp": cp}
    return json.dumps(
        {
            "fen": " ".join(board.fen().split()[:4]),
            "evals": [{"depth": 10, "pvs": [{"cp": 0}]}, {"depth": 30, "pvs": [pv, {"cp": 9}]}],
        }
    )


@pytest.fixture(scope="module")
def boards():
    return random_positions(400, seed=21)


class TestLichessConversion:
    def test_the_deepest_evaluation_is_used(self, boards):
        record = parse_record(database_line(boards[0], WHITE))
        assert record["cp"] == int(material_cp(boards[0])) and record["mate"] is None

    @pytest.mark.parametrize("line", ["", "not json", '{"fen": "x"}', '{"fen":"x","evals":[]}'])
    def test_unusable_lines_are_skipped(self, line):
        assert parse_record(line) is None

    @pytest.mark.parametrize("perspective", [WHITE, SIDE_TO_MOVE])
    def test_the_perspective_is_detected_from_the_data(self, boards, perspective):
        records = [parse_record(database_line(b, perspective)) for b in boards]
        assert detect_perspective(records) == perspective

    def test_too_little_data_is_an_error(self, boards):
        with pytest.raises(ValueError, match="not enough"):
            detect_perspective([parse_record(database_line(boards[0], WHITE))])

    @pytest.mark.parametrize("perspective", [WHITE, SIDE_TO_MOVE])
    def test_rows_are_always_from_whites_point_of_view(self, boards, perspective):
        lines = [database_line(b, perspective) for b in boards]
        rows = list(convert(lines, limit=len(lines)))
        assert len(rows) == len(boards)
        for board, row in zip(boards, rows, strict=True):
            assert row["cp"] == int(material_cp(board))

    def test_limit_is_respected_and_mates_are_converted(self, boards):
        lines = [database_line(b, WHITE) for b in boards[:300]]
        lines.append(database_line(boards[300], WHITE, mate=-3))
        rows = list(convert(lines, limit=60))
        assert len(rows) == 60
        row = to_row(parse_record(lines[-1]), WHITE)
        assert row == {"fen": row["fen"], "cp": "", "is_mate": True, "mate_distance": -3}

    def test_mate_sign_flips_with_the_perspective(self):
        record = {"fen": "8/8/8/8/8/8/8/k1K5 b - -", "cp": None, "mate": 2}
        assert to_row(record, SIDE_TO_MOVE)["mate_distance"] == -2
        assert to_row(record, WHITE)["mate_distance"] == 2

    def test_csv_roundtrips_through_the_training_loader(self, boards, tmp_path):
        from chess_engine.training.dataset import load_evaluations

        lines = [database_line(b, WHITE) for b in boards[:300]]
        path = tmp_path / "evals.csv"
        with open(path, "w", newline="") as handle:
            assert write_csv(convert(lines, 80), handle) == 80
        assert len(load_evaluations(path)) > 70


@pytest.mark.skipif(find_stockfish() is None, reason="Stockfish is not installed")
class TestStockfishLabelling:
    def test_labels_are_from_whites_point_of_view(self):
        up_a_queen = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
        rows = list(label_positions([up_a_queen], find_stockfish(), depth=6))
        assert rows[0]["cp"] > 300 and not rows[0]["is_mate"]

    def test_mates_are_reported_as_mates(self):
        rows = list(
            label_positions(["6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"], find_stockfish(), depth=6)
        )
        assert rows[0]["is_mate"] and rows[0]["mate_distance"] == 1 and rows[0]["cp"] == ""


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_random_positions_script_helper_is_reproducible():
    module = load_script("label_with_stockfish")
    assert list(module.random_fens(5, seed=3)) == list(module.random_fens(5, seed=3))


def test_plots_are_written_for_benchmarked_models(tmp_path, monkeypatch):
    monkeypatch.setattr(cards, "CARDS_DIR", tmp_path / "cards")
    for name, mae, ms in (("a", 150.0, 0.1), ("b", 100.0, 30.0)):
        cards.update_card(
            name,
            status="current",
            benchmark={"mae_cp": mae, "ms_per_eval": ms, "accuracy": 0.7, "within_100": 0.5},
        )
    module = load_script("plot_models")
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "img")
    assert module.main() == 0
    assert (tmp_path / "img" / "models_error_vs_time.png").stat().st_size > 1000
    assert (tmp_path / "img" / "models_agreement.png").stat().st_size > 1000


def test_plotting_without_results_fails_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cards, "CARDS_DIR", tmp_path / "cards")
    assert load_script("plot_models").main() == 1
    assert "No benchmarked models" in capsys.readouterr().err


def test_write_csv_header():
    handle = io.StringIO()
    write_csv([], handle)
    assert handle.getvalue().strip() == "fen,cp,is_mate,mate_distance"


def bakeoff_fixture(raw: Path, side: Path, count: int = 1500) -> None:
    """One shard with mates (every ninth position) and mixed depths, and its sidecar."""
    raw.mkdir(parents=True)
    side.mkdir(parents=True)
    records = []
    for index, board in enumerate(random_positions(count, seed=8)):
        mate = 3 if index % 9 == 0 else None
        best = next(iter(board.legal_moves), None)
        records.append(
            {
                "fen": " ".join(board.fen().split()[:4]),
                "cp": None if mate else material_cp(board),
                "mate": mate,
                "depth": 12 + 8 * (index % 2),
                "knodes": 1,
                "best_move": best.uci() if best else None,
                "n_evals": 1,
                "n_pvs": 1,
            }
        )
    write_shard(shard_path(raw, 0), records, {"perspective": "white"})
    build_sidecar(shard_path(raw, 0), side)


def test_bakeoff_depth_filter_counts_kept_training_rows(tmp_path):
    module = load_script("bakeoff_representation")
    raw, side = tmp_path / "raw", tmp_path / "side"
    bakeoff_fixture(raw, side)
    variant = next(v for v in module.VARIANTS if v.min_depth == 20)
    batches = list(module.training_batches(variant, raw, side, 100, 16, quiet=False))
    rows = sum(len(b) for b in batches)
    assert rows == 100
    assert all((b["depth"] >= 20).all() for b in batches)


def test_bakeoff_runs_on_fixture(tmp_path, monkeypatch, capsys):
    module = load_script("bakeoff_representation")
    monkeypatch.setattr(cards, "CARDS_DIR", tmp_path / "cards")
    monkeypatch.setattr(effort, "WORKERS", 2)
    raw, side = tmp_path / "raw", tmp_path / "side"
    bakeoff_fixture(raw, side)
    report, out = tmp_path / "BAKEOFF.md", tmp_path / "bakeoff.json"

    code = module.main(
        [
            "--raw", str(raw), "--sidecar", str(side),
            "--train-positions", "2000", "--test-positions", "500", "--effort", "10",
            "--tiebreak-games", "1", "--base", "10", "--increment", "0.1", "--workers", "2",
            "--tag-prefix", "test-bake",
            "--out", str(out), "--report", str(report),
        ]
    )  # fmt: skip
    assert code == 0
    data = json.loads(out.read_text())
    assert len(data["variants"]) == 8
    assert len(data["tiebreak"]["players"]) == 2
    text = report.read_text()
    assert all(variant.label in text for variant in module.VARIANTS)
    assert text.rstrip().splitlines()[-1] == "Decision: pending"
    # Temporary models and cards are deleted unless --keep-models is given.
    assert not artifact_path("pst_tapered@test-bake-0").exists()
    assert list((tmp_path / "cards").glob("*@test-bake-*.json")) == []


def test_bakeoff_refuses_to_overwrite_existing_tagged_models(tmp_path, monkeypatch, capsys):
    module = load_script("bakeoff_representation")
    monkeypatch.setattr(cards, "CARDS_DIR", tmp_path / "cards")
    monkeypatch.setattr(registry, "ARTIFACTS_DIR", tmp_path / "artifacts")
    (tmp_path / "artifacts").mkdir()
    cards.update_card("pst_tapered@test-bake-2", status="bake-off")
    raw, side = tmp_path / "raw", tmp_path / "side"
    bakeoff_fixture(raw, side, count=300)

    code = module.main(
        ["--raw", str(raw), "--sidecar", str(side), "--tag-prefix", "test-bake",
         "--out", str(tmp_path / "b.json"), "--report", str(tmp_path / "B.md")]
    )  # fmt: skip
    assert code == 2
    assert "pst_tapered@test-bake-2" in capsys.readouterr().out
    assert cards.card_path("pst_tapered@test-bake-2").exists(), "a card it did not create"
    assert not (tmp_path / "artifacts" / "pst_tapered@test-bake-0.npz").exists()
    assert not (tmp_path / "b.json").exists()
