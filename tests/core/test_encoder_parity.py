"""Parity of the core encoder (``_core.encode_many``) with its reference, ``features.py``."""

from pathlib import Path

import chess
import numpy as np
import pytest

from chess_engine import _core
from chess_lab.models.features import FEATURE_SETS

#: Edge cases first (bare kings, both castlings, nine queens, a full pawn rank, en passant), then
#: positions reached by random play; versioned so that the test does not depend on the generator.
FENS = (Path(__file__).parents[1] / "data" / "encoder_fens.txt").read_text().split("\n")[:-1]


def test_the_fen_file_has_over_a_thousand_positions():
    assert len(FENS) >= 1000


@pytest.mark.slow  # python-chess encodes each position of three sets, at ~100 us each
@pytest.mark.parametrize("name", sorted(FEATURE_SETS))
def test_the_core_encodes_every_position_like_python(name):
    rows, cols, values = _core.encode_many(FENS, name)
    assert rows.tolist() == sorted(rows.tolist())
    for row, fen in enumerate(FENS):
        indices, wanted = FEATURE_SETS[name].active(chess.Board(fen))
        at = rows == row
        assert cols[at].tolist() == indices, fen
        assert np.allclose(values[at], wanted, rtol=1e-6, atol=0), fen


def test_an_unknown_set_or_an_illegal_fen_is_a_value_error():
    with pytest.raises(ValueError):
        _core.encode_many(FENS[:1], "nothing")
    with pytest.raises(ValueError):
        _core.encode_many(["not a fen"], "pst_ctx")
