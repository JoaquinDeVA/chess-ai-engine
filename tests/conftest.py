"""Shared fixtures."""

import random
from typing import List

import chess
import pytest

from models.base import EvaluationModel
from models.features import VectorFeatures

PIECE_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def material_cp(board: chess.Board) -> float:
    """Material balance in centipawns from White's point of view."""
    return float(
        sum(
            PIECE_CP.get(p.piece_type, 0) * (1 if p.color else -1)
            for p in board.piece_map().values()
        )
    )


class MaterialStubModel(EvaluationModel):
    """Deterministic stand-in for a trained model: evaluates by counting material.

    Lets the ML bots be tested without any trained artifact.
    """

    name = "stub"
    extractor = VectorFeatures()
    artifact_suffix = ".stub"
    architecture = "material count"

    @classmethod
    def load(cls, path=None):
        return cls()

    def save(self, path=None):
        raise NotImplementedError

    def predict_target(self, inputs):
        raise NotImplementedError

    def fit(self, inputs, targets, validation=None, config=None):
        return {}

    def count_parameters(self) -> int:
        return 0

    def predict_cp(self, board: chess.Board) -> float:
        return max(-1267.0, min(1267.0, material_cp(board)))


@pytest.fixture
def stub_model() -> MaterialStubModel:
    return MaterialStubModel()


def random_positions(count: int, seed: int = 0, max_plies: int = 70) -> List[chess.Board]:
    """Positions reached by random play, reproducible through ``seed``."""
    rng = random.Random(seed)
    boards = []
    while len(boards) < count:
        board = chess.Board()
        for _ in range(rng.randint(0, max_plies)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over():
            boards.append(board)
    return boards


@pytest.fixture(scope="session")
def sample_boards() -> List[chess.Board]:
    return random_positions(60, seed=123)


MATE_IN_ONE = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"  # 1.Ra8#
