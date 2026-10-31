"""The trained models as the laboratory defines them: weights, feature set and saved file.

A model is its weights plus the name of the feature set the Rust core computes. The laboratory
trains it and saves it as a ``.npz`` file in ``chess_engine.models.ARTIFACTS_DIR``; the app loads
that file straight into the core (:func:`chess_engine.models.load_evaluator`) without these classes.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from chess_engine.models import artifact_path, split_spec
from chess_engine.models.loader import DEFAULT_PERSPECTIVE

#: Centipawns that correspond to a normalized target of ``1.0``. It is the 0.5% quantile of the
#: course dataset: evaluations beyond it are clipped, so the target covers ~99% of the data.
CP_SCALE = 1267.0

Layer = Tuple[np.ndarray, np.ndarray]


class SparseModel(ABC):
    """A model over the sparse features of ``feature_set``, saved in one ``.npz`` file.

    Models work on a *normalized target* ``y`` in ``[-1, 1]`` (centipawns divided by
    :data:`CP_SCALE`, from White's point of view).

    :param rep: Representation of the training target as a plain dictionary; only its
        ``perspective`` is read here and the rest is saved untouched.
    :cvar name: Model name, also used for the artifact and card file names.
    :cvar feature_set: Name of the feature set, one of ``_core.FEATURE_SETS``.
    :cvar architecture: One-line description of the architecture, for the model card.
    """

    name: ClassVar[str]
    feature_set: ClassVar[str]
    architecture: ClassVar[str] = ""

    def __init__(self, rep: Optional[Dict[str, Any]] = None):
        self.rep: Dict[str, Any] = dict(rep) if rep else {}

    @property
    def perspective(self) -> str:
        """``"white"`` or ``"side_to_move"``: whose point of view the score has."""
        return self.rep.get("perspective", DEFAULT_PERSPECTIVE)

    @classmethod
    def default_path(cls) -> Path:
        """Location of the default saved model of this class.

        :return: ``ARTIFACTS_DIR/<name>.npz``.
        """
        return artifact_path(cls.name)

    @classmethod
    def _check_features(cls, meta: Dict[str, Any], path: Path) -> None:
        """Reject a file saved with the feature set of another model."""
        if meta["features"] != cls.feature_set:
            raise ValueError(f"{path} was saved with features {meta['features']!r}")

    @classmethod
    @abstractmethod
    def load(cls, path: Optional[Path] = None) -> "SparseModel":
        """Load a saved model.

        :param path: File to load; defaults to :meth:`default_path`.
        :return: The model, with the representation it was trained with.
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If the file was saved with another feature set.
        """

    @abstractmethod
    def save(self, path: Optional[Path] = None) -> Path:
        """Save the model.

        :param path: Destination; defaults to :meth:`default_path`.
        :return: The path that was written.
        """

    @abstractmethod
    def count_parameters(self) -> int:
        """Number of trainable parameters.

        :return: Parameter count.
        """


def _open_for_loading(cls: Type[SparseModel], path: Optional[Path]) -> Path:
    path = Path(path) if path else cls.default_path()
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}")
    return path


class LinearEvalModel(SparseModel):
    """Logistic regression over the sparse features of ``feature_set``.

    :param weights: Coefficients, one per feature of the set under the model's perspective.
    :param bias: Intercept.
    :param rep: Representation of the training target as a plain dictionary.
    """

    def __init__(
        self,
        weights: np.ndarray,
        bias: float = 0.0,
        rep: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(rep)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)

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
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If the file was saved with another feature set.
        """
        path = _open_for_loading(cls, path)
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            cls._check_features(meta, path)
            weights, bias = data["weights"], float(data["bias"])
        return cls(weights=weights, bias=bias, rep=meta["rep"])

    def save(self, path: Optional[Path] = None) -> Path:
        """Save weights, bias, feature set and representation in one ``.npz`` file.

        :param path: Destination; defaults to :meth:`default_path`.
        :return: The path that was written.
        """
        meta = {"features": self.feature_set, "rep": self.rep}
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


class NeuralEvalModel(SparseModel):
    """Feed-forward network over sparse inputs, with weights in NumPy arrays.

    The first weight layer is an accumulator over the slotted features.

    :param layers: ``(weights, bias)`` per weight layer, with shapes ``(inputs, hidden[0])``,
        ``(hidden[0], hidden[1])``, ..., ``(hidden[-1], 1)``.
    :param rep: Representation of the training target as a plain dictionary.
    :cvar hidden: Sizes of the hidden layers, the first being the accumulator.
    """

    hidden: ClassVar[Tuple[int, ...]] = ()

    def __init__(self, layers: Sequence[Layer], rep: Optional[Dict[str, Any]] = None):
        super().__init__(rep)
        self.layers: List[Layer] = [
            (np.asarray(weights, dtype=np.float64), np.asarray(bias, dtype=np.float64))
            for weights, bias in layers
        ]

    def count_parameters(self) -> int:
        """Weights and biases of all layers.

        :return: Total number of parameters.
        """
        return sum(weights.size + bias.size for weights, bias in self.layers)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "NeuralEvalModel":
        """Load a saved model.

        The size of the input layer is not checked here: the core rejects a mismatch with its
        feature set when it builds the evaluation.

        :param path: File to load; defaults to :meth:`default_path`.
        :return: The model, with the representation it was trained with.
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If the file was saved with another feature set or hidden sizes, or
            its layers do not chain.
        """
        path = _open_for_loading(cls, path)
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            cls._check_features(meta, path)
            if tuple(meta["hidden"]) != cls.hidden:
                raise ValueError(
                    f"{path} was saved with hidden sizes {tuple(meta['hidden'])}, "
                    f"expected {cls.hidden}"
                )
            layers = [(data[f"W{i}"], data[f"b{i}"]) for i in range(len(cls.hidden) + 1)]
        fan_outs = (*cls.hidden, 1)
        for position, ((weights, bias), fan_out) in enumerate(zip(layers, fan_outs, strict=True)):
            if weights.shape[1] != fan_out or bias.shape != (fan_out,):
                raise ValueError(
                    f"{path}: layer {position} has shapes {weights.shape} and {bias.shape}, "
                    f"expected {fan_out} outputs"
                )
        return cls(layers=layers, rep=meta["rep"])

    def save(self, path: Optional[Path] = None) -> Path:
        """Save the layers, the feature set, the hidden sizes and the representation.

        :param path: Destination; defaults to :meth:`default_path`.
        :return: The path that was written.
        """
        path = Path(path) if path else self.default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        for position, (weights, bias) in enumerate(self.layers):
            arrays[f"W{position}"], arrays[f"b{position}"] = weights, bias
        meta = {"features": self.feature_set, "rep": self.rep, "hidden": list(self.hidden)}
        # A file object keeps numpy from appending ".npz" to the requested name.
        with open(path, "wb") as handle:
            np.savez(handle, meta=np.array(json.dumps(meta)), **arrays)
        return path


class PstTaperedModel(LinearEvalModel):
    """Tapered piece-square tables (768 weights) with a logistic output."""

    name = "linear_pst"
    feature_set = "linear_pst"
    architecture = "Tapered piece-square tables (768 weights), logistic"


class PstTermsModel(LinearEvalModel):
    """Tapered piece-square tables plus eleven classical terms (790 weights), logistic."""

    name = "linear_pst_terms"
    feature_set = "linear_pst_terms"
    architecture = "Tapered piece-square tables plus 11 classical terms (790 weights), logistic"


class MaterialLinearModel(LinearEvalModel):
    """Logistic regression on five piece-count differences (and the turn under ``"white"``)."""

    name = "linear_material"
    feature_set = "material5"
    architecture = "Logistic regression on 5 material counts (own minus opponent)"


class KingBucketModel(LinearEvalModel):
    """King-bucketed piece-square features (10,368 weights), logistic."""

    name = "linear_king_bucket"
    feature_set = "king_bucket"
    architecture = "King-bucketed piece-square features (10,368 weights), logistic"


class Nn4Model(NeuralEvalModel):
    """768 slotted piece-square inputs, a 256 accumulator, 32, 32 and one output (4 layers)."""

    name = "net4_pst"
    feature_set = "pst_slotted"
    hidden = (256, 32, 32)
    architecture = "Sparse network: 768 slotted piece-square inputs, 256-32-32-1 (4 layers)"


#: The models of the project by name.
MODELS: Dict[str, Type[SparseModel]] = {
    cls.name: cls
    for cls in (PstTaperedModel, PstTermsModel, MaterialLinearModel, KingBucketModel, Nn4Model)
}


def available_models() -> List[str]:
    """Names of the models the laboratory defines.

    :return: Sorted list of names.
    """
    return sorted(MODELS)


def get_model_class(name: str) -> Type[SparseModel]:
    """Look a model class up by name.

    :param name: Model name.
    :return: The model class.
    :raises KeyError: With the list of valid names if ``name`` is unknown.
    """
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}; available: {', '.join(available_models())}"
        ) from None


def load_model(spec: str, path: Optional[Path] = None) -> SparseModel:
    """Load the saved model named by ``spec``.

    :param spec: Model name, or ``name@tag`` for a tagged artifact.
    :param path: Saved file; defaults to :func:`chess_engine.models.artifact_path`.
    :return: The loaded model.
    :raises KeyError: If the name is unknown.
    :raises FileNotFoundError: If the model file does not exist (train it first).
    """
    name, _ = split_spec(spec)
    return get_model_class(name).load(path or artifact_path(spec))
