"""Abstract interface of the saved position evaluators."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Optional

#: Centipawns that correspond to a normalized target of ``1.0``. It is the 0.5% quantile of the
#: course dataset: evaluations beyond it are clipped, so the target covers ~99% of the data.
CP_SCALE = 1267.0

#: Default folder of the trained model files.
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


class EvaluationModel(ABC):
    """A trained model that estimates the evaluation of chess positions.

    Subclasses define how the model is stored; the core runs it
    (:func:`chess_engine.models.native.model_evaluator`) and the laboratory trains it.
    Models work on a *normalized target* ``y`` in ``[-1, 1]`` (centipawns divided by
    :data:`CP_SCALE`, from White's point of view).

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
