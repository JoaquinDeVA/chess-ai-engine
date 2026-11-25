"""Tests of the models table and the history file."""

from chess_lab.measure import report
from chess_lab.models.store import cards
from chess_lab.models.target import Representation
from tests.conftest import ACCURACY


def test_report_renders_the_columns(isolated_cards):
    representation = Representation().to_dict()
    cards.update_card(
        "net_pst_w64_both",
        status="current",
        architecture="PST",
        parameters=1234,
        training={"effort": 40, "representation": representation},
        accuracy=ACCURACY,
        cost={"us_per_eval": 12.5},
    )
    text = report.render_tables()
    header = (
        "| Model | Params | Representation | Effort | Spearman | MAE | Excess (nats) | "
        "Best move, all / non-quiet | µs per eval |"
    )
    row = (
        f"| `net_pst_w64_both` | 1,234 | {Representation().name} | 40 | 0.912 | 0.083 | "
        "0.0123 | 34.5% / 28.0% | 12.5 |"
    )
    assert header in text and row in text


def test_a_model_trained_twice_shows_only_its_highest_effort(isolated_cards):
    for effort in (30, 100):
        cards.update_card(
            f"m@e{effort}", status="current", architecture="A", training={"effort": effort}
        )
    lines = [line for line in report.render_tables().splitlines() if line.startswith("| `m`")]
    assert len(lines) == 1 and "| 100 |" in lines[0]


def test_models_without_sections_show_dashes(isolated_cards):
    cards.update_card("fresh", status="current", architecture="A")
    assert f"| `fresh` |{' — |' * 8}" in report.render_tables()


def test_history_file_is_updated_between_markers_only(isolated_cards):
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


def test_history_file_without_markers_is_left_alone(isolated_cards):
    report.HISTORY_FILE.write_text("no markers here")
    assert not report.update_history_file()
    assert report.HISTORY_FILE.read_text() == "no markers here"
    assert not report.update_history_file(isolated_cards / "does-not-exist.md")
