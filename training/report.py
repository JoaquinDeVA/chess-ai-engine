"""Render the model cards into the tables of ``MODELS_HISTORY.md``.

The file is hand-written except for the block between the ``BEGIN``/``END`` markers, which
this module regenerates: ``python -m training.report``.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .cards import PROJECT_ROOT, load_all_cards

HISTORY_FILE = PROJECT_ROOT / "MODELS_HISTORY.md"
BEGIN_MARKER = "<!-- BEGIN GENERATED TABLES -->"
END_MARKER = "<!-- END GENERATED TABLES -->"

_STATUS_ORDER = {"current": 0, "legacy": 1}
_DASH = "—"


def _percent(value: Optional[float]) -> str:
    """Format a share in ``[0, 1]`` as a percentage."""
    return _DASH if value is None else f"{100 * value:.1f}%"


def _number(value: Optional[float], digits: int = 1) -> str:
    """Format a number, or a dash when it is missing."""
    return _DASH if value is None else f"{value:,.{digits}f}"


def _size(card: Dict[str, Any]) -> str:
    """Format the on-disk size of a model."""
    kb = card.get("size_kb")
    if kb is None:
        return _DASH
    return f"{kb / 1024:.1f} MB" if kb >= 1024 else f"{kb:.0f} KB"


def _strength(card: Dict[str, Any]) -> str:
    """Format the match results as ``opponent W-D-L`` pairs."""
    strength = card.get("strength")
    if not strength:
        return _DASH
    pairs = "; ".join(f"{opponent} {score}" for opponent, score in strength["results"].items())
    return f"{pairs} (depth {strength['depth']})"


def _training_data(card: Dict[str, Any]) -> str:
    """Summarize the training set and epochs."""
    training = card.get("training") or {}
    if not training:
        return _DASH
    parts = []
    if training.get("train_samples"):
        parts.append(f"{training['train_samples']:,} positions")
    if training.get("dataset"):
        parts.append(str(training["dataset"]))
    if training.get("epochs"):
        parts.append(f"{training['epochs']} epochs")
    return ", ".join(parts) or _DASH


def render_benchmarked(cards: List[Dict[str, Any]]) -> str:
    """Table of models that have weights (current and legacy).

    :param cards: All model cards.
    :return: Markdown table.
    """
    rows = [card for card in cards if card.get("status") in _STATUS_ORDER]
    rows.sort(key=lambda card: (_STATUS_ORDER[card["status"]], card["name"]))
    lines = [
        "| Model | Status | Architecture | Params | Size | Training data | Time / eval | "
        "MAE (cp) | RMSE (cp) | Agreement | ±50 cp | ±100 cp | Playing strength |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for card in rows:
        bench = card.get("benchmark") or {}
        time_per_eval = bench.get("ms_per_eval")
        lines.append(
            "| `{name}` | {status} | {arch} | {params} | {size} | {data} | {ms} | {mae} | {rmse} "
            "| {acc} | {w50} | {w100} | {strength} |".format(
                name=card["name"],
                status=card["status"],
                arch=card.get("architecture", _DASH),
                params=_number(card.get("parameters"), 0),
                size=_size(card),
                data=_training_data(card),
                ms=_DASH if time_per_eval is None else f"{time_per_eval:.2f} ms",
                mae=_number(bench.get("mae_cp")),
                rmse=_number(bench.get("rmse_cp")),
                acc=_percent(bench.get("accuracy")),
                w50=_percent(bench.get("within_50")),
                w100=_percent(bench.get("within_100")),
                strength=_strength(card),
            )
        )
    return "\n".join(lines)


def render_historical(cards: List[Dict[str, Any]]) -> str:
    """Table of experiments whose weights were not kept.

    :param cards: All model cards.
    :return: Markdown table.
    """
    rows = sorted(
        (card for card in cards if card.get("status") == "historical"),
        key=lambda card: card.get("order", 0),
    )
    lines = [
        "| Experiment | Source | Architecture | Epochs | Logged val. MAE (normalized) | Notes |",
        "|---|---|---|---:|---:|---|",
    ]
    for card in rows:
        training = card.get("training") or {}
        lines.append(
            "| {name} | {source} | {arch} | {epochs} | {mae} | {notes} |".format(
                name=card.get("display_name", card["name"]),
                source=card.get("source", _DASH),
                arch=card.get("architecture", _DASH),
                epochs=training.get("epochs", _DASH),
                mae=_number(card.get("logged_val_mae"), 3),
                notes=card.get("notes", ""),
            )
        )
    return "\n".join(lines)


def render_tables(cards: Optional[List[Dict[str, Any]]] = None) -> str:
    """Both generated tables, with their headings.

    :param cards: Model cards (default: all cards on disk).
    :return: Markdown, without the markers.
    """
    cards = load_all_cards() if cards is None else cards
    return (
        "### Models with weights\n\n"
        f"{render_benchmarked(cards)}\n\n"
        "### Historical experiments (weights not kept)\n\n"
        f"{render_historical(cards)}"
    )


def update_history_file(path: Path = HISTORY_FILE) -> bool:
    """Replace the generated block of ``MODELS_HISTORY.md``.

    :param path: File to update.
    :return: True if the file was rewritten, False if it does not exist or has no markers yet.
    """
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if BEGIN_MARKER not in text or END_MARKER not in text:
        return False
    head, rest = text.split(BEGIN_MARKER, 1)
    _, tail = rest.split(END_MARKER, 1)
    path.write_text(f"{head}{BEGIN_MARKER}\n{render_tables()}\n{END_MARKER}{tail}", encoding="utf-8")
    return True


if __name__ == "__main__":
    print("updated" if update_history_file() else f"nothing to update in {HISTORY_FILE}")
