"""Linear models as the app sees them: weights read from a file and evaluated by the core.

A model here is its weights, its bias and the name of the feature set; the core computes the
features and the score (:mod:`chess_engine.models.native`). Building, training and evaluating a
model in Python is the job of the laboratory.
"""

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import numpy as np

from .base import EvaluationModel
from .registry import register_model

#: Point of view of the score when the saved representation does not say.
DEFAULT_PERSPECTIVE = "side_to_move"


def _feature_set_of(cls: type) -> str:
    """Name of the feature set of a concrete model class.

    :param cls: A class derived from :class:`LinearEvalModel`.
    :return: Its ``feature_set`` attribute.
    :raises TypeError: If the class is the abstract base, which defines no feature set.
    """
    feature_set = getattr(cls, "feature_set", None)
    if feature_set is None:
        raise TypeError(f"{cls.__name__} defines no feature set; use a concrete model")
    return feature_set


class LinearEvalModel(EvaluationModel):
    """Logistic regression over the sparse features of ``feature_set``.

    :param weights: Coefficients, one per feature of the set under the model's perspective.
    :param bias: Intercept.
    :param rep: Representation of the training target as a plain dictionary; the app only reads
        its ``perspective`` and saves the rest untouched.
    :cvar feature_set: Name of the feature set, one of ``_core.FEATURE_SETS``.
    :cvar artifact_suffix: ``.npz``.
    """

    feature_set: ClassVar[str]
    artifact_suffix = ".npz"

    def __init__(
        self,
        weights: np.ndarray,
        bias: float = 0.0,
        rep: Optional[Dict[str, Any]] = None,
    ):
        self.rep: Dict[str, Any] = dict(rep) if rep else {}
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)

    @property
    def perspective(self) -> str:
        """``"white"`` or ``"side_to_move"``: whose point of view the score has."""
        return self.rep.get("perspective", DEFAULT_PERSPECTIVE)

    def count_parameters(self) -> int:
        """Coefficients plus the intercept.

        :return: ``size + 1``.
        """
        return int(self.weights.size) + 1

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LinearEvalModel":
        """Load a saved model.

        The number of weights is not checked here: the core rejects a mismatch with its feature
        set when it builds the evaluation.

        :param path: File to load; defaults to :meth:`default_path`.
        :return: The model, with the representation it was trained with.
        :raises TypeError: If called on the abstract base, which has no feature set.
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If the file was saved with another feature set.
        """
        feature_set = _feature_set_of(cls)
        path = Path(path) if path else cls.default_path()
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            weights, bias = data["weights"], float(data["bias"])
        if meta["features"] != feature_set:
            raise ValueError(f"{path} was saved with features {meta['features']!r}")
        return cls(weights=weights, bias=bias, rep=meta["rep"])

    def save(self, path: Optional[Path] = None) -> Path:
        """Save weights, bias, feature set and representation in one ``.npz`` file.

        :param path: Destination; defaults to :meth:`default_path`.
        :return: The path that was written.
        :raises TypeError: If the model has no feature set (the abstract base class).
        """
        meta = {"features": _feature_set_of(type(self)), "rep": self.rep}
        path = Path(path) if path else self.default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # A file object keeps numpy from appending ".npz" to the requested name.
        with open(path, "wb") as handle:
            np.savez(
                handle,
                weights=self.weights,
                bias=np.float64(self.bias),
                meta=np.array(json.dumps(meta)),
            )
        return path


@register_model("linear_pst")
class PstTaperedModel(LinearEvalModel):
    """Tapered piece-square tables (768 weights) with a logistic output."""

    feature_set = "linear_pst"
    architecture = "Tapered piece-square tables (768 weights), logistic"


@register_model("linear_pst_terms")
class PstTermsModel(LinearEvalModel):
    """Tapered piece-square tables plus eleven classical terms (790 weights), logistic."""

    feature_set = "linear_pst_terms"
    architecture = "Tapered piece-square tables plus 11 classical terms (790 weights), logistic"


@register_model("linear_material")
class MaterialLinearModel(LinearEvalModel):
    """Logistic regression on five piece-count differences (and the turn under ``"white"``)."""

    feature_set = "material5"
    architecture = "Logistic regression on 5 material counts (own minus opponent)"


@register_model("linear_king_bucket")
class KingBucketModel(LinearEvalModel):
    """King-bucketed piece-square features (10,368 weights), logistic."""

    feature_set = "king_bucket"
    architecture = "King-bucketed piece-square features (10,368 weights), logistic"
