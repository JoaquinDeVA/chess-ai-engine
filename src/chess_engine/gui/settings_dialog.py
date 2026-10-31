"""Dialog where the user tunes one AI player."""

from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from chess_engine import _core
from chess_engine.gui.settings import MAX_SETTABLE_DEPTH, AISettings


class AISettingsDialog(QDialog):
    """Edit an :class:`AISettings`; read the result with :meth:`settings` after ``exec``."""

    def __init__(self, current: AISettings, title: str = "AI settings", parent=None):
        """Fill the controls with the current values.

        :param current: Settings to start from.
        :param title: Window title.
        :param parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.depth = QSpinBox()
        self.depth.setRange(0, MAX_SETTABLE_DEPTH)
        self.depth.setSpecialValueText("No limit")
        self.depth.setValue(current.max_depth or 0)
        self.depth.setToolTip("No limit: the time decides")

        self.fixed_time = QCheckBox("Fixed time per move")
        self.fixed_time.setChecked(current.move_time is not None)
        self.move_time = QDoubleSpinBox()
        self.move_time.setRange(0.1, 600.0)
        self.move_time.setSingleStep(0.5)
        self.move_time.setSuffix(" s")
        self.move_time.setValue(current.move_time or 3.0)
        self.move_time.setEnabled(self.fixed_time.isChecked())
        self.fixed_time.toggled.connect(self.move_time.setEnabled)

        self.book = QCheckBox("Opening book")
        self.book.setChecked(current.opening_book)
        self.tablebase = QCheckBox("Endgame tablebases (up to 5 pieces)")
        self.tablebase.setChecked(current.tablebase)

        self.threads = QSpinBox()
        self.threads.setRange(1, 64)
        self.threads.setValue(current.threads)
        self.techniques = {}
        advanced = QGroupBox("Advanced")
        advanced_form = QFormLayout(advanced)
        advanced_form.addRow("Search threads", self.threads)
        for technique in _core.TECHNIQUES:
            box = QCheckBox(technique)
            box.setChecked(technique in current.techniques)
            self.techniques[technique] = box
            advanced_form.addRow(box)
        advanced.setVisible(False)
        toggle = QPushButton("Advanced…")
        toggle.setCheckable(True)
        toggle.toggled.connect(advanced.setVisible)
        toggle.toggled.connect(lambda _: self.adjustSize())

        form = QFormLayout()
        form.addRow("Maximum depth", self.depth)
        form.addRow(self.fixed_time, self.move_time)
        form.addRow(self.book)
        form.addRow(self.tablebase)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(toggle)
        layout.addWidget(advanced)
        layout.addWidget(buttons)

    def settings(self) -> AISettings:
        """The settings the controls show."""
        return AISettings(
            max_depth=self.depth.value() or None,
            move_time=self.move_time.value() if self.fixed_time.isChecked() else None,
            opening_book=self.book.isChecked(),
            tablebase=self.tablebase.isChecked(),
            techniques=tuple(name for name, box in self.techniques.items() if box.isChecked()),
            threads=self.threads.value(),
        )


def edit_settings(current: AISettings, title: str, parent=None) -> Optional[AISettings]:
    """Show the dialog and return the new settings, or None if the user cancelled.

    :param current: Settings to start from.
    :param title: Window title.
    :param parent: Parent widget.
    :return: Chosen settings or None.
    """
    dialog = AISettingsDialog(current, title, parent)
    return dialog.settings() if dialog.exec() == QDialog.Accepted else None
