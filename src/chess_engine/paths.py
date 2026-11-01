"""Where the repository is: the app and the laboratory find their files from here."""

from pathlib import Path

#: Root of the repository (the folder with ``rust/``, ``data/`` and ``src/``).
ROOT = Path(__file__).resolve().parents[2]
