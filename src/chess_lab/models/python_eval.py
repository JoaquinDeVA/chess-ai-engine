"""Evaluation of a saved model in Python, for the measurements of the laboratory.

The game never uses it: the core evaluates the models. Here it serves to judge a model against
the labelled positions, to time it and to compare it with the core.
"""

from collections.abc import Iterable

import chess
import numpy as np

from .definitions import CP_SCALE, NeuralEvalModel, SparseModel
from .linear_features import features_of
from .representation import Representation, score_to_cp


def _sigmoid(z: float) -> float:
    """Logistic function, stable for large magnitudes.

    :param z: Logit.
    :return: ``1 / (1 + exp(-z))``.
    """
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class PythonEvaluator:
    """A saved linear model or network evaluated in Python.

    :param model: The model to evaluate.
    """

    def __init__(self, model: SparseModel):
        self.model = model
        self.name = model.name
        self.architecture = model.architecture
        self.rep = Representation.from_dict(model.rep)
        self.features = features_of(model)
        if not isinstance(model, NeuralEvalModel):
            # Plain Python floats make the per-position sum in :meth:`logit` fast.
            self._weights = model.weights.tolist()

    def count_parameters(self) -> int:
        """Parameters of the model.

        :return: The model's own count.
        """
        return self.model.count_parameters()

    def logit(self, board: chess.Board) -> float:
        """Score of a position before the logistic function.

        It is the logit of the side to move under ``"side_to_move"``, of White under ``"white"``.

        :param board: The position.
        :return: ``w . x + b`` for a linear model, the output of the last layer for a network.
        """
        indices, values = self.features.active(board, self.rep.perspective)
        if isinstance(self.model, NeuralEvalModel):
            layers = self.model.layers
            weights, bias = layers[0]
            hidden = np.clip(np.asarray(values) @ weights[indices] + bias, 0.0, 1.0)
            for weights, bias in layers[1:-1]:
                hidden = np.maximum(hidden @ weights + bias, 0.0)
            weights, bias = layers[-1]
            return float(hidden @ weights[:, 0] + bias[0])
        total = self.model.bias
        for index, value in zip(indices, values, strict=True):
            total += self._weights[index] * value
        return total

    def expected_score(self, board: chess.Board) -> float:
        """Expected score of White in ``[0, 1]``.

        :param board: The position.
        :return: The logistic score, flipped to White's point of view when Black is to move
            under ``"side_to_move"``.
        """
        score = float(_sigmoid(self.logit(board)))
        if self.rep.perspective == "side_to_move" and board.turn == chess.BLACK:
            return 1.0 - score
        return score

    def predict_cp(self, board: chess.Board) -> float:
        """Evaluate one position.

        :param board: Position to evaluate.
        :return: Centipawns from White's point of view, clipped to ``[-CP_SCALE, CP_SCALE]``.
        """
        cp = float(score_to_cp(self.expected_score(board), self.rep))
        return min(CP_SCALE, max(-CP_SCALE, cp))

    def predict_cp_batch(self, boards: Iterable[chess.Board]) -> np.ndarray:
        """Evaluate many positions.

        :param boards: Positions to evaluate.
        :return: Centipawns from White's point of view, shape ``(n,)``.
        """
        return np.array([self.predict_cp(board) for board in boards], dtype=np.float64)
