"""Abstract interface of the machine-learning position evaluators."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Iterable, Optional

import chess
import numpy as np

#: Centipawns that correspond to a normalized target of ``1.0``. It is the 0.5% quantile of the
#: course dataset: evaluations beyond it are clipped, so the target covers ~99% of the data.
CP_SCALE = 1267.0

#: Default folder of the trained model files.
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

#: Evaluations returned to the search never get closer than this to the terminal values 0 and 1.
EXPECTED_SCORE_MARGIN = 1e-2


def cp_to_expected_score(cp: float, margin: float = EXPECTED_SCORE_MARGIN) -> float:
    """Map centipawns (White's point of view) to an expected score in ``(0, 1)``.

    ``0`` centipawns is ``0.5``; ``+CP_SCALE`` or more is ``1 - margin``. The linear map keeps
    the scale of the heuristic evaluations, which also live in ``[margin, 1 - margin]``.

    :param cp: Evaluation in centipawns, positive when White is better.
    :param margin: Distance kept to the terminal values.
    :return: Expected score for White in ``[margin, 1 - margin]``.
    """
    score = 0.5 + 0.5 * float(np.clip(cp / CP_SCALE, -1.0, 1.0))
    return min(1.0 - margin, max(margin, score))


class EvaluationModel(ABC):
    """A trained model that estimates the evaluation of chess positions.

    Subclasses define how a network or estimator is stored, run and trained. Models work on a
    *normalized target* ``y`` in ``[-1, 1]`` (centipawns divided by :data:`CP_SCALE`, from
    White's point of view); :meth:`predict_cp` converts it back to centipawns.

    :cvar name: Registry key, set by :func:`models.registry.register_model`.
    :cvar artifact_suffix: File extension of the saved model.
    :cvar architecture: One-line description of the architecture, for the model card.
    """

    name: ClassVar[str]
    artifact_suffix: ClassVar[str]
    architecture: ClassVar[str] = ""

    @classmethod
    def default_path(cls) -> Path:
        """Location of the default saved model of this class.

        :return: ``models/artifacts/<name><suffix>``.
        """
        return ARTIFACTS_DIR / f"{cls.name}{cls.artifact_suffix}"

    @classmethod
    @abstractmethod
    def load(cls, path: Optional[Path] = None) -> "EvaluationModel":
        """Load a saved model.

        :param path: File to load; defaults to :meth:`default_path`.
        :return: The ready-to-use model.
        :raises FileNotFoundError: If the file does not exist.
        """

    @abstractmethod
    def save(self, path: Optional[Path] = None) -> Path:
        """Save the model.

        :param path: Destination; defaults to :meth:`default_path`.
        :return: The path that was written.
        """

    @abstractmethod
    def count_parameters(self) -> int:
        """Number of trainable parameters (coefficients for linear models).

        :return: Parameter count.
        """

    @abstractmethod
    def predict_cp(self, board: chess.Board) -> float:
        """Evaluate one position.

        :param board: Position to evaluate.
        :return: Centipawns from White's point of view, within ``[-CP_SCALE, CP_SCALE]``.
        """

    def expected_score(self, board: chess.Board) -> float:
        """Expected score of White in ``[0, 1]``.

        :param board: Position to evaluate.
        :return: Expected score for White in ``[0, 1]``; by default the centipawns of
            :meth:`predict_cp` mapped by :func:`cp_to_expected_score`.
        """
        return cp_to_expected_score(self.predict_cp(board))

    @abstractmethod
    def predict_cp_batch(self, boards: Iterable[chess.Board]) -> np.ndarray:
        """Evaluate many positions at once.

        :param boards: Positions to evaluate.
        :return: Centipawns from White's point of view, shape ``(n,)``.
        """
