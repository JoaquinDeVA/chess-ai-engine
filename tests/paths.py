"""Locations shared by the tests, independent of how deep a test file sits."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).resolve().parent / "golden"
SYZYGY = Path(__file__).resolve().parent / "data" / "syzygy"
