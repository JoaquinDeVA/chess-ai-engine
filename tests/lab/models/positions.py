"""Positions and raw-shard rows shared by the model tests."""

import chess
import numpy as np
import pandas as pd

from tests.conftest import material_cp

WHITE_UP_A_QUEEN = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_UP_A_QUEEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"


def frame(boards: list[chess.Board]) -> pd.DataFrame:
    """Raw-shard-like rows whose White-POV centipawns are the material balance."""
    return pd.DataFrame(
        {
            "fen": [board.fen() for board in boards],
            "cp": [material_cp(board) for board in boards],
            "mate": np.full(len(boards), np.nan),
            "best_move": [None] * len(boards),
            "quiet": [True] * len(boards),
            "white_to_move": [board.turn == chess.WHITE for board in boards],
        }
    )
