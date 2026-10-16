#!/usr/bin/env python3
"""Render the tactical-pattern diagrams used by ``docs/TACTICAL_PATTERNS.md`` as SVG files.

python scripts/render_pattern_boards.py
"""

from pathlib import Path

import chess
import chess.svg

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"

# name -> (FEN, [(from, to), ...] arrows to draw)
DIAGRAMS = {
    "absolute_pin": ("4k3/4n3/8/8/8/8/8/4R1K1 w - - 0 1", [("e1", "e7"), ("e7", "e8")]),
    "relative_pin": ("3qk3/8/3n4/8/8/8/8/3RK3 w - - 0 1", [("d1", "d6"), ("d6", "d8")]),
    "fork": ("2r1k3/8/3N4/8/8/8/8/4K3 b - - 0 1", [("d6", "e8"), ("d6", "c8")]),
    "skewer": ("r3k2R/8/8/8/8/8/8/4K3 b - - 0 1", [("h8", "e8"), ("e8", "a8")]),
}


def render(name: str, fen: str, arrows) -> Path:
    """Write one diagram.

    :param name: File name without extension.
    :param fen: Position to draw.
    :param arrows: ``(from_square, to_square)`` name pairs to draw as arrows.
    :return: Path of the SVG written.
    """
    board = chess.Board(fen)
    svg = chess.svg.board(
        board,
        size=320,
        arrows=[
            chess.svg.Arrow(chess.parse_square(a), chess.parse_square(b), color="#d9480fcc")
            for a, b in arrows
        ],
    )
    path = OUTPUT_DIR / f"{name}.svg"
    path.write_text(svg, encoding="utf-8")
    return path


def main() -> int:
    """Render every diagram.

    :return: Process exit code.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (fen, arrows) in DIAGRAMS.items():
        print(render(name, fen, arrows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
