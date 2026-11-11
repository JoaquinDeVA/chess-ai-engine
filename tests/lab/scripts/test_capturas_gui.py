"""Tests of src/chess_lab/scripts/capturas_gui.py: one non-empty PNG per view, drawn offscreen."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from chess_lab.scripts import capturas_gui  # noqa: E402


def test_saves_one_png_per_view(tmp_path):
    capturas_gui.main(["--out", str(tmp_path)])
    names = sorted(p.name for p in tmp_path.glob("*.png"))
    assert names == ["gui-game.png", "gui-settings.png", "gui-setup.png"]
    assert all((tmp_path / n).stat().st_size > 1000 for n in names)
