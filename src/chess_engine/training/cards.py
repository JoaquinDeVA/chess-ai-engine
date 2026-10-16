"""Model cards: small JSON files with everything known about a model.

Cards live in ``models/cards/<name>.json``. ``training.train`` and ``training.evaluate`` fill
them in; ``training.report`` renders them into ``MODELS_HISTORY.md``.
"""

import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

CARDS_DIR = Path(__file__).resolve().parent.parent / "models" / "cards"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def card_path(name: str) -> Path:
    """Path of the card of model ``name``.

    :param name: Model name.
    :return: ``models/cards/<name>.json``.
    """
    return CARDS_DIR / f"{name}.json"


def load_card(name: str) -> Dict[str, Any]:
    """Read the card of a model, or an empty skeleton if it has none yet.

    :param name: Model name.
    :return: The card as a dictionary.
    """
    path = card_path(name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"name": name}


def save_card(name: str, card: Dict[str, Any]) -> Path:
    """Write a card to disk (pretty-printed, keys in insertion order).

    :param name: Model name.
    :param card: Card contents.
    :return: The path that was written.
    """
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    path = card_path(name)
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def update_card(name: str, **sections: Any) -> Dict[str, Any]:
    """Merge ``sections`` into the card of ``name`` and save it.

    Dictionary sections are merged key by key; other values replace the old ones.

    :param name: Model name.
    :param sections: Top-level card entries to set.
    :return: The updated card.
    """
    card = load_card(name)
    for key, value in sections.items():
        if isinstance(value, dict) and isinstance(card.get(key), dict):
            card[key].update(value)
        else:
            card[key] = value
    save_card(name, card)
    return card


def load_all_cards() -> List[Dict[str, Any]]:
    """Read every card.

    :return: Cards sorted by name.
    """
    if not CARDS_DIR.exists():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(CARDS_DIR.glob("*.json"))
    ]


def current_commit() -> str:
    """Short hash of the code revision, with ``+dirty`` if the working tree has changes.

    :return: The hash, or ``"unknown"`` outside a git checkout.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def today() -> str:
    """Today's date in ISO format.

    :return: ``YYYY-MM-DD``.
    """
    return date.today().isoformat()
