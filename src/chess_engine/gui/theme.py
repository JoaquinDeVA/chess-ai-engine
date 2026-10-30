"""Colours and style sheet of the window, after the dark look and green board of chess.com."""

BOARD_LIGHT = "#eeeed2"
BOARD_DARK = "#769656"
HIGHLIGHT = (255, 255, 51, 105)  # last move and selected square, over the square
BACKGROUND = "#302e2b"
PANEL = "#262421"
PANEL_LIGHT = "#3c3a37"
TEXT = "#ffffff"
TEXT_MUTED = "#bababa"
ACCENT = "#81b64c"
ACCENT_HOVER = "#95c35f"
DANGER = "#c33a3a"

STYLESHEET = f"""
QWidget {{ background: {BACKGROUND}; color: {TEXT}; font-size: 14px; }}
QLabel {{ background: transparent; }}
QFrame#panel {{ background: {PANEL}; border-radius: 6px; }}
QFrame#playerbar {{ background: transparent; }}
QLabel#clock {{ background: {PANEL}; color: {TEXT_MUTED}; border-radius: 4px; padding: 4px 12px;
    font-size: 24px; font-weight: bold; font-family: "DejaVu Sans Mono", monospace; }}
QLabel#clock[active="true"] {{ background: {TEXT}; color: {PANEL}; }}
QLabel#clock[low="true"] {{ color: {DANGER}; }}
QLabel#clock[active="true"][low="true"] {{ background: {DANGER}; color: {TEXT}; }}
QLabel#playername {{ font-weight: bold; font-size: 15px; }}
QLabel#banner {{ background: {PANEL_LIGHT}; border-radius: 6px; padding: 10px; font-size: 16px;
    font-weight: bold; }}
QLabel#title {{ font-size: 22px; font-weight: bold; }}
QLabel#muted {{ color: {TEXT_MUTED}; }}
QPushButton {{ background: {PANEL_LIGHT}; border: none; border-radius: 6px; padding: 8px 14px;
    font-weight: bold; }}
QPushButton:hover {{ background: #4a4845; }}
QPushButton:disabled {{ color: #777; background: {PANEL}; }}
QPushButton#primary {{ background: {ACCENT}; color: {TEXT}; font-size: 17px; padding: 12px 20px; }}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#danger:hover {{ background: {DANGER}; }}
QTableWidget {{ background: {PANEL}; border: none; gridline-color: transparent;
    selection-background-color: {PANEL_LIGHT}; alternate-background-color: #2b2926; }}
QTableWidget::item {{ padding: 3px 8px; }}
QHeaderView {{ background: transparent; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {PANEL}; border: 1px solid #4a4845;
    border-radius: 4px; padding: 4px 8px; min-height: 22px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; selection-background-color: {ACCENT}; }}
QGroupBox {{ border: 1px solid #4a4845; border-radius: 6px; margin-top: 14px; padding-top: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {TEXT_MUTED}; }}
QCheckBox {{ spacing: 8px; }}
QMenu {{ background: {PANEL}; border: 1px solid #4a4845; }}
QMenu::item:selected {{ background: {ACCENT}; }}
"""
