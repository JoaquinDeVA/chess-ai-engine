"""Locations shared by the tests, independent of how deep a test file sits."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSITIONS = Path(__file__).resolve().parent / "data" / "positions.json"
SYZYGY = Path(__file__).resolve().parent / "data" / "syzygy"
