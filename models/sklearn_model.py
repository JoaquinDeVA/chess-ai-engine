"""Adapter for scikit-learn estimators (stored with joblib)."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import EvaluationModel


class SklearnModel(EvaluationModel):
    """Evaluation model backed by a scikit-learn regressor or pipeline.

    Subclasses set ``extractor`` and implement :meth:`build_estimator`. Only the first array
    of the extractor is fed to the estimator, so use a single-input extractor.

    :param estimator: A fitted (or to be fitted) scikit-learn regressor.
    """

    artifact_suffix = ".pkl"

    def __init__(self, estimator: Any):
        self.estimator = estimator

    @classmethod
    def build_estimator(cls, config: Dict[str, Any]) -> Any:
        """Create an untrained estimator.

        :param config: Hyper-parameters from the training configuration.
        :return: A scikit-learn regressor with ``fit`` and ``predict``.
        """
        raise NotImplementedError(f"{cls.__name__} does not define build_estimator()")

    @classmethod
    def new(cls, config: Optional[Dict[str, Any]] = None) -> "SklearnModel":
        """Create an untrained model.

        :param config: Hyper-parameters passed to :meth:`build_estimator`.
        :return: A model ready for :meth:`fit`.
        """
        return cls(cls.build_estimator(config or {}))

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SklearnModel":
        """Load a pickled estimator.

        :param path: File to load; defaults to :meth:`default_path`.
        :return: The loaded model.
        :raises FileNotFoundError: If the file does not exist.
        """
        import joblib

        path = Path(path) if path else cls.default_path()
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        return cls(joblib.load(path))

    def save(self, path: Optional[Path] = None) -> Path:
        """Pickle the estimator.

        :param path: Destination; defaults to :meth:`default_path`.
        :return: The path that was written.
        """
        import joblib

        path = Path(path) if path else self.default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path)
        return path

    def _predict_raw(self, inputs: Tuple[np.ndarray, ...]) -> np.ndarray:
        """Run the estimator.

        :param inputs: Encoded batch; only the first array is used.
        :return: Predictions, shape ``(n,)``.
        """
        return np.asarray(self.estimator.predict(inputs[0]), dtype=np.float64).ravel()

    def fit(
        self,
        inputs: Tuple[np.ndarray, ...],
        targets: np.ndarray,
        validation: Optional[Tuple[Tuple[np.ndarray, ...], np.ndarray]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fit the estimator (``validation`` and ``config`` are not used).

        :param inputs: Encoded training positions.
        :param targets: Normalized targets.
        :param validation: Ignored; accepted for interface compatibility.
        :param config: Ignored; hyper-parameters are applied in :meth:`build_estimator`.
        :return: Summary with the number of training samples.
        """
        self.estimator.fit(inputs[0], targets)
        return {"epochs": 1, "train_samples": int(len(targets))}

    def count_parameters(self) -> int:
        """Count the coefficients and intercepts of the final estimator.

        :return: Number of fitted parameters, ``0`` if the estimator exposes none.
        """
        final = self.estimator
        if hasattr(final, "steps"):  # Pipeline
            final = final.steps[-1][1]
        total = 0
        for attribute in ("coef_", "intercept_"):
            if hasattr(final, attribute):
                total += int(np.size(getattr(final, attribute)))
        return total
