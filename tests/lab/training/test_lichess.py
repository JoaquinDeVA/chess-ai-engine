"""Detection of the perspective (white or side to move) of the Lichess evaluations."""

import chess
import pytest

from chess_lab.training.lichess_raw import SIDE_TO_MOVE, WHITE, detect_perspective
from tests.conftest import material_cp, random_positions


def records_of(boards: list[chess.Board], perspective: str) -> list[dict]:
    """Fake records whose evaluation follows material, from the given perspective."""
    records = []
    for board in boards:
        cp = int(material_cp(board))
        if perspective == SIDE_TO_MOVE and board.turn == chess.BLACK:
            cp = -cp
        records.append({"fen": " ".join(board.fen().split()[:4]), "cp": cp, "mate": None})
    return records


@pytest.fixture(scope="module")
def boards():
    return random_positions(400, seed=21)


class TestPerspectiveDetection:
    @pytest.mark.parametrize("perspective", [WHITE, SIDE_TO_MOVE])
    def test_the_perspective_is_detected_from_the_data(self, boards, perspective):
        assert detect_perspective(records_of(boards, perspective)) == perspective

    def test_too_little_data_is_an_error(self, boards):
        with pytest.raises(ValueError, match="not enough"):
            detect_perspective(records_of(boards[:1], WHITE))
