"""Render the model cards into the table of ``MODELS_HISTORY.md``.

The file is hand-written except for the block between the ``BEGIN``/``END`` markers, which
this module regenerates: ``python -m chess_engine.comparison.report``.

Each row reads the card sections ``accuracy`` and ``cost`` (written by ``comparison.evaluate``),
``tradeoff`` (depth and search agreement per time per move, from the profiler) and ``tournament``
(score and Elo, from the round-robin). A missing section shows a dash.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import chess_engine.models  # noqa: F401  (loads models first: avoids an import cycle)
from chess_engine.models.cards import PROJECT_ROOT, load_all_cards
from chess_engine.training.representation import Representation

HISTORY_FILE = PROJECT_ROOT / "docs" / "training" / "MODELS_HISTORY.md"
BEGIN_MARKER = "<!-- BEGIN GENERATED TABLES -->"
END_MARKER = "<!-- END GENERATED TABLES -->"

#: Models that are not retrained nor played: only their accuracy and cost are measured.
ACCURACY_RECORDS = frozenset({"ridge_features", "mlp_features", "cnn_light", "cnn_resnet"})

#: Times per move (seconds) of the depth columns, as keys of the ``tradeoff`` section.
DEPTH_SECONDS = ("1", "3", "10")

#: Time per move (seconds) of the search agreement column.
AGREEMENT_SECONDS = "10"

_STATUS_ORDER = {"current": 0}
_DASH = "—"
_HEADER = (
    "| Model | Kind | Params | Representation | Effort | Spearman | MAE | "
    "Best move, all / non-quiet | µs per eval | Depth 1 s | Depth 3 s | Depth 10 s | "
    "Search agreement, 10 s | Score | Elo ± interval |"
)
_ALIGNMENT = "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


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
    """Short name of the target representation of a linear model."""
    fields = _dig(card, "training", "representation")
    return _DASH if not fields else Representation.from_dict(fields).name


def _elo(tournament: Dict[str, Any]) -> str:
    """Elo estimate with its 95% half-width, e.g. ``+43 ± 24``."""
    elo, interval = tournament.get("elo"), tournament.get("elo_interval")
    if elo is None:
        return _DASH
    return f"{elo:+.0f}" if interval is None else f"{elo:+.0f} ± {interval:.0f}"


def _row(card: Dict[str, Any]) -> str:
    """One line of the history table.

    :param card: Model card.
    :return: Markdown table row.
    """
    name = card["name"]
    accuracy = _dig(card, "accuracy", "all") or {}
    non_quiet = _dig(card, "accuracy", "non_quiet") or {}
    tournament = card.get("tournament") or {}
    tradeoff = card.get("tradeoff") or {}
    best_move = (
        f"{_percent(accuracy.get('best_move_agreement'))} / "
        f"{_percent(non_quiet.get('best_move_agreement'))}"
        if accuracy
        else _DASH
    )
    effort = _dig(card, "training", "effort")
    cells = [
        f"`{name}`",
        "accuracy record" if name in ACCURACY_RECORDS else "plays",
        _number(card.get("parameters"), 0),
        _representation(card),
        _DASH if effort is None else str(effort),
        _number(accuracy.get("spearman"), 3),
        _number(accuracy.get("mae"), 3),
        best_move,
        _number(_dig(card, "cost", "us_per_eval")),
    ]
    cells += [_number(_dig(tradeoff, seconds, "mean_depth")) for seconds in DEPTH_SECONDS]
    cells.append(_percent(_dig(tradeoff, AGREEMENT_SECONDS, "search_agreement")))
    score = tournament.get("score")
    cells.append(_DASH if score is None else f"{score:.1f}%")
    cells.append(_elo(tournament))
    return "| " + " | ".join(cells) + " |"


def render_tables(cards: Optional[List[Dict[str, Any]]] = None) -> str:
    """The comparison table of all models.

    Models with status ``current`` are listed; the models that play come first.

    :param cards: Model cards (default: all cards on disk).
    :return: Markdown, without the markers.
    """
    rows = [
        card
        for card in (load_all_cards() if cards is None else cards)
        if card.get("status") in _STATUS_ORDER
    ]
    rows.sort(key=lambda card: (card["name"] in ACCURACY_RECORDS, card["name"]))
    return "\n".join([_HEADER, _ALIGNMENT, *(_row(card) for card in rows)])


def update_history_file(path: Optional[Path] = None) -> bool:
    """Replace the generated block of ``MODELS_HISTORY.md``.

    :param path: File to update (default: ``MODELS_HISTORY.md`` of the project).
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
