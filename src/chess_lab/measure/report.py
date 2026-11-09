"""Render the model cards into the table of ``RESULTS.md``.

The file is hand-written except for the block between the ``BEGIN``/``END`` markers, which
this module regenerates: ``python -m chess_lab.measure.report``.

Each row reads the card sections ``accuracy`` and ``cost``, written by ``measure.evaluate``.
A missing value shows a dash.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from chess_engine.paths import ROOT
from chess_lab.models.store.cards import load_all_cards
from chess_lab.models.target import Representation

HISTORY_FILE = ROOT / "docs" / "RESULTS.md"
BEGIN_MARKER = "<!-- BEGIN GENERATED TABLES -->"
END_MARKER = "<!-- END GENERATED TABLES -->"

_STATUS_ORDER = {"current": 0, "reference": 1, "retired": 2}
_DASH = "—"
_HEADER = (
    "| Model | Params | Representation | Effort | Spearman | MAE | "
    "Best move, all / non-quiet | µs per eval |"
)
_ALIGNMENT = "|---|---:|---|---:|---:|---:|---:|---:|"


def _dig(card: Dict[str, Any], *keys: str) -> Any:
    """Follow nested keys of a card; None as soon as one is missing.

    :param card: Model card.
    :param keys: Keys to follow.
    :return: The value found, or None.
    """
    value: Any = card
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _percent(value: Optional[float]) -> str:
    """Format a share in ``[0, 1]`` as a percentage."""
    return _DASH if value is None else f"{100 * value:.1f}%"


def _number(value: Optional[float], digits: int = 1) -> str:
    """Format a number, or a dash when it is missing."""
    return _DASH if value is None else f"{value:,.{digits}f}"


def _representation(card: Dict[str, Any]) -> str:
    """Short name of the target representation of a model."""
    fields = _dig(card, "training", "representation")
    return _DASH if not fields else Representation.from_dict(fields).name


def _row(card: Dict[str, Any]) -> str:
    """One line of the history table.

    :param card: Model card.
    :return: Markdown table row.
    """
    name = f"`{card['name'].split('@')[0]}`"
    if card.get("status") == "retired":
        name += " (retired)"
    accuracy = _dig(card, "accuracy", "all") or {}
    non_quiet = _dig(card, "accuracy", "non_quiet") or {}
    best_move = (
        f"{_percent(accuracy.get('best_move_agreement'))} / "
        f"{_percent(non_quiet.get('best_move_agreement'))}"
        if accuracy
        else _DASH
    )
    effort = _dig(card, "training", "effort")
    cells = [
        name,
        _number(card.get("parameters"), 0),
        _representation(card),
        _DASH if effort is None else str(effort),
        _number(accuracy.get("spearman"), 3),
        _number(accuracy.get("mae"), 3),
        best_move,
        _number(_dig(card, "cost", "us_per_eval")),
    ]
    return "| " + " | ".join(cells) + " |"


def render_tables(cards: Optional[List[Dict[str, Any]]] = None) -> str:
    """The comparison table of all models.

    Models with status ``current``, ``reference`` (the heuristic ``h1``) or ``retired`` are listed,
    in that order and by name within each status; a retired model is marked as such.

    :param cards: Model cards (default: all cards on disk).
    :return: Markdown, without the markers.
    """
    rows = [
        card
        for card in (load_all_cards() if cards is None else cards)
        if card.get("status") in _STATUS_ORDER
    ]
    rows.sort(key=lambda card: (_STATUS_ORDER[card["status"]], card["name"].split("@")[0]))
    return "\n".join([_HEADER, _ALIGNMENT, *(_row(card) for card in rows)])


def update_history_file(path: Optional[Path] = None) -> bool:
    """Replace the generated block of ``RESULTS.md``.

    :param path: File to update (default: ``RESULTS.md`` of the project).
    :return: True if the file was rewritten, False if it does not exist or has no markers yet.
    """
    path = HISTORY_FILE if path is None else path
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if BEGIN_MARKER not in text or END_MARKER not in text:
        return False
    head, rest = text.split(BEGIN_MARKER, 1)
    _, tail = rest.split(END_MARKER, 1)
    path.write_text(
        f"{head}{BEGIN_MARKER}\n{render_tables()}\n{END_MARKER}{tail}", encoding="utf-8"
    )
    return True


if __name__ == "__main__":
    print("updated" if update_history_file() else f"nothing to update in {HISTORY_FILE}")
