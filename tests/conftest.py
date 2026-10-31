"""Shared fixtures."""

import random
from typing import List

import chess
import numpy as np
import pytest

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


class MaterialStubModel:
    """Deterministic stand-in for a trained model: evaluates by counting material.

    Python only, so it does not play (the core cannot run it): it serves the judge and the cards.
    """

    name = "stub"
    architecture = "material count"

    def count_parameters(self) -> int:
        return 0

    def predict_cp(self, board: chess.Board) -> float:
        return max(-1267.0, min(1267.0, material_cp(board)))

    def predict_cp_batch(self, boards):
        return np.array([self.predict_cp(board) for board in boards], dtype=np.float64)


@pytest.fixture
def stub_model() -> MaterialStubModel:
    return MaterialStubModel()


@pytest.fixture(scope="session")
def material_model():
    """Name of a model the core can play: the shipped material logistic model."""
    return "linear_material"


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


#: Shipped model artifacts that no test may overwrite.
PROTECTED_ARTIFACTS = ("linear_pst_terms.npz", "net4_pst.npz")
_PROTECTED_HASHES: dict = {}


def _artifact_hashes() -> dict:
    import hashlib

    from chess_engine.models import ARTIFACTS_DIR

    hashes = {}
    for name in PROTECTED_ARTIFACTS:
        path = ARTIFACTS_DIR / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return hashes


def pytest_sessionstart(session):
    _PROTECTED_HASHES.update(_artifact_hashes())


def pytest_sessionfinish(session, exitstatus):
    changed = [n for n, h in _artifact_hashes().items() if h != _PROTECTED_HASHES.get(n)]
    if changed:
        print(f"\nERROR: a test overwrote shipped artifacts: {changed}")
        session.exitstatus = 1
