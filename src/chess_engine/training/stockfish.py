"""Label positions with a local Stockfish engine (UCI)."""

import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import chess
import chess.engine

#: Places where Debian/Ubuntu put the binary, which are often not in ``PATH``.
COMMON_LOCATIONS = ("/usr/games/stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish")


def find_stockfish() -> str | None:
    """Locate a Stockfish executable.

    :return: Path of the executable, or None if it is not installed.
    """
    found = shutil.which("stockfish")
    if found:
        return found
    return next((path for path in COMMON_LOCATIONS if Path(path).exists()), None)


def label_positions(
    fens: Iterable[str], engine_path: str, depth: int = 12
) -> Iterator[dict[str, Any]]:
    """Evaluate positions with Stockfish, from White's point of view.

    :param fens: Positions to evaluate.
    :param engine_path: Stockfish executable.
    :param depth: Search depth per position.
    :return: Rows with ``fen, cp, is_mate, mate_distance`` (the project's CSV columns).
    """
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for fen in fens:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].white()
            mate = score.mate()
            yield {
                "fen": " ".join(board.fen().split()[:4]),
                "cp": "" if mate is not None else score.score(),
                "is_mate": mate is not None,
                "mate_distance": "" if mate is None else mate,
            }
