"""Logistic models over sparse board features: fast, non-neural evaluators.

The model predicts a score in ``[0, 1]`` as ``sigmoid(w . x + b)``, where ``x`` are the sparse
features of :mod:`linear_features`. Its target is the :class:`Representation` of the training
data: the score of the side to move (or of White), converted back to centipawns when the model
answers :meth:`predict_cp`.
"""

import json
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import chess
import numpy as np
import pandas as pd

from chess_engine.training.representation import Representation, score_to_cp, target

from .base import CP_SCALE, EvaluationModel
from .linear_features import LinearFeatures, PstFeatures
from .registry import register_model

#: Hyper-parameters used when the training configuration leaves a key out.
DEFAULT_FIT_CONFIG: Dict[str, Any] = {
    "l2": 1e-6,
    "lr": 0.01,
    "batch_size": 4096,
    "max_epochs": 20,
    "patience": 3,
}
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPS = 1e-8


def _sigmoid(z: float) -> float:
    """Logistic function, stable for large magnitudes.

    :param z: Logit.
    :return: ``1 / (1 + exp(-z))``.
    """
    return 0.5 * (1.0 + math.tanh(0.5 * z))


class LinearEvalModel(EvaluationModel):
    """Logistic regression over the sparse features of ``features``.

    :param weights: Coefficients, shape ``(features.size(rep.perspective),)``; zeros if omitted.
    :param bias: Intercept.
    :param rep: Representation of the target; defaults to :class:`Representation`.
    :cvar features: Sparse encoding of positions.
    :cvar artifact_suffix: ``.npz``.
    """

    features: ClassVar[LinearFeatures]
    artifact_suffix = ".npz"

    def __init__(
        self,
        weights: Optional[np.ndarray] = None,
        bias: float = 0.0,
        rep: Optional[Representation] = None,
    ):
        self.rep = rep if rep is not None else Representation()
        size = self.features.size(self.rep.perspective)
        self.weights = np.zeros(size) if weights is None else weights
        self.bias = float(bias)

    @property
    def weights(self) -> np.ndarray:
        """Coefficients as a numpy array, shape ``(size,)``."""
        return self._weights

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        self._weights = np.asarray(value, dtype=np.float64)
        # Plain Python floats make the per-position sum in :meth:`logit` fast.
        self._weights_list: List[float] = self._weights.tolist()

    def logit(self, board: chess.Board) -> float:
        """Linear score of a position, before the logistic function.

        It is the logit of the side to move under ``"side_to_move"``, of White under ``"white"``.

        :param board: The position.
        :return: ``w . x + b``, computed in pure Python.
        """
        indices, values = self.features.active(board, self.rep.perspective)
        weights = self._weights_list
        total = self.bias
        for index, value in zip(indices, values, strict=True):
            total += weights[index] * value
        return total

    def expected_score(self, board: chess.Board) -> float:
        """Expected score of White in ``[0, 1]``.

        :param board: The position.
        :return: The logistic score, flipped to White's point of view when Black is to move
            under ``"side_to_move"``.
        """
        score = _sigmoid(self.logit(board))
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

    def predict_target(self, inputs: Tuple[np.ndarray, ...]) -> np.ndarray:
        """Not available: linear models work on boards, see :meth:`expected_score`.

        :raises NotImplementedError: Always.
        """
        raise NotImplementedError("linear models evaluate boards: use expected_score()")

    def fit(self, inputs, targets, validation=None, config=None) -> Dict[str, Any]:
        """Not available: use :meth:`fit_stream` with DataFrames of raw shard rows.

        :raises NotImplementedError: Always.
        """
        raise NotImplementedError("linear models are trained with fit_stream()")

    def count_parameters(self) -> int:
        """Coefficients plus the intercept.

        :return: ``size + 1``.
        """
        return int(self._weights.size) + 1

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LinearEvalModel":
        """Load a saved model.

        :param path: File to load; defaults to :meth:`default_path`.
        :return: The model, with the representation it was trained with.
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If the file was saved with other features.
        """
        path = Path(path) if path else cls.default_path()
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            weights, bias = data["weights"], float(data["bias"])
        if meta["features"] != cls.features.name:
            raise ValueError(f"{path} was saved with features {meta['features']!r}")
        return cls(weights=weights, bias=bias, rep=Representation.from_dict(meta["rep"]))

    def save(self, path: Optional[Path] = None) -> Path:
        """Save weights, bias, features name and representation in one ``.npz`` file.

        :param path: Destination; defaults to :meth:`default_path`.
        :return: The path that was written.
        """
        path = Path(path) if path else self.default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"features": self.features.name, "rep": self.rep.to_dict()}
        # A file object keeps numpy from appending ".npz" to the requested name.
        with open(path, "wb") as handle:
            np.savez(
                handle,
                weights=self._weights,
                bias=np.float64(self.bias),
                meta=np.array(json.dumps(meta)),
            )
        return path

    def fit_stream(
        self,
        batches: Iterable[pd.DataFrame],
        validation: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Train with Adam on the binary cross-entropy against soft targets.

        ``batches`` is read once per epoch, so it must be re-iterable (a list, or an object whose
        ``__iter__`` restarts), not a one-shot generator. The weights of the epoch with the lowest
        validation loss are kept.

        :param batches: DataFrames with columns ``fen, cp, mate, white_to_move``.
        :param validation: Same columns, used for early stopping.
        :param config: Keys ``l2``, ``lr``, ``batch_size``, ``max_epochs``, ``patience``; missing
            keys take :data:`DEFAULT_FIT_CONFIG`.
        :return: Summary with ``epochs``, ``best_epoch``, ``train_losses``, ``val_losses``,
            ``train_rows``, ``seconds`` and the ``config`` used.
        :raises ValueError: If an epoch yields no rows.
        """
        cfg = {**DEFAULT_FIT_CONFIG, **config}
        started = time.perf_counter()
        size = self._weights.size
        theta = np.append(self._weights, self.bias)  # coefficients, then the intercept
        moment, velocity = np.zeros_like(theta), np.zeros_like(theta)
        step = 0
        validation_data = self._encode(validation)
        best_theta, best_loss, best_epoch = theta.copy(), math.inf, 0
        train_losses: List[float] = []
        val_losses: List[float] = []
        bad_epochs = 0
        train_rows = 0

        for epoch in range(1, int(cfg["max_epochs"]) + 1):
            loss_sum, rows_seen = 0.0, 0
            for frame in batches:
                for local in _minibatches(*self._encode(frame), int(cfg["batch_size"])):
                    local_rows, local_cols, local_vals, local_y = local
                    count = len(local_y)
                    z = _scores(local_rows, local_cols, local_vals, count, theta)
                    loss_sum += float(np.sum(_losses(z, local_y)))
                    rows_seen += count
                    grad_z = 0.5 * (1.0 + np.tanh(0.5 * z)) - local_y
                    grad_w = (
                        np.bincount(
                            local_cols,
                            weights=local_vals * grad_z[local_rows],
                            minlength=size,
                        )
                        / count
                        + float(cfg["l2"]) * theta[:size]
                    )
                    grad = np.append(grad_w, grad_z.mean())
                    step += 1
                    moment = _ADAM_BETA1 * moment + (1 - _ADAM_BETA1) * grad
                    velocity = _ADAM_BETA2 * velocity + (1 - _ADAM_BETA2) * grad * grad
                    m_hat = moment / (1 - _ADAM_BETA1**step)
                    v_hat = velocity / (1 - _ADAM_BETA2**step)
                    theta -= float(cfg["lr"]) * m_hat / (np.sqrt(v_hat) + _ADAM_EPS)
            if rows_seen == 0:
                raise ValueError("batches yielded no rows; pass a re-iterable collection")
            train_rows = rows_seen
            train_losses.append(loss_sum / rows_seen)

            val_rows, val_cols, val_vals, val_y = validation_data
            val_z = _scores(val_rows, val_cols, val_vals, len(val_y), theta)
            val_loss = float(np.mean(_losses(val_z, val_y)))
            val_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss, best_epoch, best_theta = val_loss, epoch, theta.copy()
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs > int(cfg["patience"]):
                    break

        self.weights = best_theta[:size]
        self.bias = float(best_theta[size])
        return {
            "epochs": len(train_losses),
            "best_epoch": best_epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_rows": train_rows,
            "seconds": round(time.perf_counter() - started, 3),
            "config": cfg,
        }

    def _encode(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sparse features of the rows of a frame, row-ordered, with their soft targets.

        :param frame: Rows with ``fen, cp, mate, white_to_move``.
        :return: ``(rows, cols, values, targets)``; ``rows`` is sorted and indexes ``targets``.
        """
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for row, fen in enumerate(frame["fen"]):
            indices, values = self.features.active(chess.Board(fen), self.rep.perspective)
            rows.extend([row] * len(indices))
            cols.extend(indices)
            vals.extend(values)
        y = target(
            np.asarray(frame["cp"], dtype=np.float64),
            np.asarray(frame["mate"], dtype=np.float64),
            np.asarray(frame["white_to_move"], dtype=bool),
            self.rep,
        )
        return (
            np.asarray(rows, dtype=np.intp),
            np.asarray(cols, dtype=np.intp),
            np.asarray(vals, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
        )


def _minibatches(rows, cols, vals, y, batch_size: int):
    """Split encoded rows into consecutive minibatches.

    :param rows: Sorted row indices of the sparse entries.
    :param cols: Feature index of each entry.
    :param vals: Value of each entry.
    :param y: Targets, one per row.
    :param batch_size: Rows per minibatch.
    :return: Generator of ``(local_rows, cols, vals, targets)`` with rows renumbered from 0.
    """
    for start in range(0, len(y), batch_size):
        stop = min(len(y), start + batch_size)
        first, last = np.searchsorted(rows, [start, stop])
        yield rows[first:last] - start, cols[first:last], vals[first:last], y[start:stop]


def _scores(rows, cols, vals, count: int, theta: np.ndarray) -> np.ndarray:
    """Logits of ``count`` rows, from the sparse entries and ``theta = (weights, bias)``.

    :return: Array of shape ``(count,)``.
    """
    weights = theta[:-1]
    return np.bincount(rows, weights=vals * weights[cols], minlength=count) + theta[-1]


def _losses(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Binary cross-entropy with soft targets, from logits.

    :return: ``log(1 + exp(z)) - y * z`` per row.
    """
    return np.logaddexp(0.0, z) - y * z


@register_model("pst_tapered")
class PstTaperedModel(LinearEvalModel):
    """Tapered piece-square tables (768 weights) with a logistic output."""

    features = PstFeatures()
    architecture = "Tapered piece-square tables (768 weights), logistic"
