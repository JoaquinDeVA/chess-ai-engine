"""Play chess against the engine, or watch two engines play, in a window.

    python main.py

Who plays each colour, the time control and the settings of every AI are chosen on the first page.
Games between engines are played with fastchess and the UCI binary (``rust/target/release/uci``).
Needs the ``gui`` extra: ``pip install -e ".[gui]"``.
"""

import sys


def main() -> int:
    """Open the GUI.

    :return: Exit code.
    """
    try:
        from chess_engine.gui.app import main as run
    except ImportError as error:
        print(f"The window needs PySide6 ({error}). Install it with: pip install -e '.[gui]'")
        return 1
    return run()


if __name__ == "__main__":
    sys.exit(main())
