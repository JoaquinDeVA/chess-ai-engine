"""Tests of the models table and the history file."""

from chess_lab.measure import evaluate, report
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
        training={
            "effort": 40,
            "date": "2026-11-20",
            "effort_config": {"positions": 29_280_737, "max_epochs": 40},
            "representation": representation,
        },
        accuracy=ACCURACY,
        cost={"us_per_eval": 12.5},
    )
    text = report.render_tables()
    header = (
        "| Model | Params | Representation | Effort | Positions (M) | Max epochs | Spearman | MAE | "
        "Excess (nats) | Best move, all / non-quiet | µs per eval |"
    )
    row = (
        f"| `net_pst_w64_both` | 1,234 | {Representation().name} | 40 | 29.3 | 40 | 0.912 | 0.083 | "
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
    assert f"| `fresh` |{' — |' * 10}" in report.render_tables()


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


def _judged_on(name, sample_hash):
    sample = {} if sample_hash is None else {"hash": sample_hash}
    cards.update_card(
        name, status="current", architecture="A", accuracy=ACCURACY | {"sample": sample}
    )


def test_rows_judged_on_another_sample_are_marked(isolated_cards):
    for name in ("a", "b", "c"):
        _judged_on(name, "same")
    _judged_on("d", "other")
    _judged_on("e", None)
    text = report.render_tables()
    rows = {
        line.split("|")[1].strip(): line for line in text.splitlines() if line.startswith("| `")
    }
    assert [name for name, line in rows.items() if "†" in name] == ["`d` †", "`e` †"]
    assert "† Judged on another test sample" in text


def test_nothing_is_marked_while_no_card_records_its_sample(isolated_cards):
    _judged_on("a", None)
    _judged_on("b", None)
    text = report.render_tables()
    assert "†" not in text


def test_the_sample_hash_follows_the_sample_order_and_content():
    fens = ["8/8/8/8/8/8/8/K6k w - - 0 1", "8/8/8/8/8/8/8/k6K w - - 0 1"]
    assert evaluate.sample_hash(fens) == evaluate.sample_hash(iter(fens))
    assert evaluate.sample_hash(fens) != evaluate.sample_hash(fens[::-1])
    assert evaluate.sample_hash(fens) != evaluate.sample_hash(fens[:1])
    assert len(evaluate.sample_hash(fens)) == 16


def test_at_equal_effort_the_plain_model_takes_the_row_whatever_the_order(isolated_cards):
    cards.update_card(
        "m@e15-warm-lr1e-4", status="current", architecture="A", training={"effort": 15}
    )
    cards.update_card(
        "m@e15", status="current", architecture="A", parameters=7, training={"effort": 15}
    )
    cards.update_card("m@e15-zz", status="current", architecture="A", training={"effort": 15})
    lines = [line for line in report.render_tables().splitlines() if line.startswith("| `m`")]
    assert len(lines) == 1 and "| 7 |" in lines[0]
    # the card list is sorted by name, so feed the variant last as well
    both = [cards.load_card("m@e15"), cards.load_card("m@e15-zz")]
    for order in (both, both[::-1]):
        (line,) = [x for x in report.render_tables(order).splitlines() if x.startswith("| `m`")]
        assert "| 7 |" in line


def test_old_effort_is_marked_and_the_real_positions_are_shown(isolated_cards):
    cards.update_card(
        "old@e30",
        status="current",
        architecture="A",
        training={"effort": 30, "date": "2026-11-14", "effort_config": {"positions": 30_000_000}},
    )
    cards.update_card(
        "new@e15",
        status="current",
        architecture="A",
        training={"effort": 15, "date": "2026-11-20", "recipe": {"positions": 29_280_737}},
    )
    text = report.render_tables()
    old, new = (
        next(x for x in text.splitlines() if x.startswith(f"| `{n}`")) for n in ("old", "new")
    )
    assert "| 30 ‡ | 30.0 |" in old and "‡" not in new
    assert "| 15 | 29.3 |" in new  # positions fall back to the recipe
    assert "‡ Effort of the old meaning" in text
