"""The trained models as the laboratory defines them: weights, feature set and saved file.

A model is a network: its layers plus the name of the feature set the Rust core computes. The
laboratory trains it, saves it as a ``.npz`` file in :data:`chess_lab.models.store.loader.ARTIFACTS_DIR`
and reads it back here, the only reader of that format; :meth:`NeuralEvalModel.evaluator` hands it
to the core.
"""

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from chess_engine import _core

from .store.loader import artifact_path, split_spec

#: Centipawns that correspond to a normalized target of ``1.0``. It is the 0.5% quantile of the
#: course dataset: evaluations beyond it are clipped, so the target covers ~99% of the data.
CP_SCALE = 1267.0

Layer = Tuple[np.ndarray, np.ndarray]


class NeuralEvalModel:
    """Feed-forward network over the sparse inputs of ``feature_set``, saved in one ``.npz`` file.

    The first weight layer is the accumulator over the sparse inputs; the output is the logit of
    the expected score of the player to move.

    :param layers: ``(weights, bias)`` per weight layer, with shapes ``(inputs, hidden[0])``,
        ``(hidden[0], hidden[1])``, ..., ``(hidden[-1], 1)``.
    :param rep: Representation of the training target as a plain dictionary, saved untouched.
    :cvar name: Model name, also used for the artifact and card file names.
    :cvar feature_set: Name of the feature set, one of ``_core.FEATURE_SETS``.
    :cvar architecture: One-line description of the architecture, for the model card.
    :cvar hidden: Sizes of the hidden layers, the first being the accumulator.
    :cvar perspective: Whether the second layer reads the accumulators of both sides (the mover's,
        then the rival's, over the same first-layer weights): its input is ``2 * hidden[0]`` wide.
    """

    name: ClassVar[str]
    feature_set: ClassVar[str]
    architecture: ClassVar[str] = ""
    hidden: ClassVar[Tuple[int, ...]] = ()
    perspective: ClassVar[bool] = False

    @classmethod
    def layer_shapes(cls) -> List[Tuple[int, int]]:
        """``(fan_in, fan_out)`` of every weight layer, accumulator first.

        :return: One pair per layer; with both perspectives the second layer has twice the inputs.
        """
        from .features import FEATURE_SETS

        sizes = (FEATURE_SETS[cls.feature_set].size(), *cls.hidden, 1)
        shapes = list(zip(sizes[:-1], sizes[1:], strict=True))
        if cls.perspective:
            shapes[1] = (2 * shapes[1][0], shapes[1][1])
        return shapes

    def __init__(self, layers: Sequence[Layer], rep: Optional[Dict[str, Any]] = None):
        self.rep: Dict[str, Any] = dict(rep) if rep else {}
        self.layers: List[Layer] = [
            (np.asarray(weights, dtype=np.float64), np.asarray(bias, dtype=np.float64))
            for weights, bias in layers
        ]

    @classmethod
    def default_path(cls) -> Path:
        """Location of the default saved model of this class.

        :return: ``ARTIFACTS_DIR/<name>.npz``.
        """
        return artifact_path(cls.name)

    def count_parameters(self) -> int:
        """Weights and biases of all layers.

        :return: Total number of parameters.
        """
        return sum(weights.size + bias.size for weights, bias in self.layers)

    def evaluator(self) -> "_core.Evaluator":
        """The model as an evaluation of the core.

        :return: Evaluator that runs inside the core.
        :raises ValueError: If the input layer does not fit the feature set.
        """
        layers = [
            (np.ascontiguousarray(weights), np.ascontiguousarray(bias))
            for weights, bias in self.layers
        ]
        return _core.Evaluator.network(self.feature_set, layers, self.perspective)

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
        path = Path(path) if path else cls.default_path()
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            if meta["features"] != cls.feature_set:
                raise ValueError(f"{path} was saved with features {meta['features']!r}")
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
        if cls.perspective and layers[1][0].shape[0] != 2 * cls.hidden[0]:
            raise ValueError(f"{path}: the second layer must read both perspectives")
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


class NnW32Model(NeuralEvalModel):
    """768 slotted piece-square inputs, a 32 accumulator and one output (2 layers)."""

    name = "net_pst_w32"
    feature_set = "pst_slotted"
    hidden = (32,)
    architecture = "Sparse network: 768 slotted piece-square inputs, 32-1 (2 layers)"


class NnW64Model(NeuralEvalModel):
    """768 slotted piece-square inputs, a 64 accumulator and one output (2 layers)."""

    name = "net_pst_w64"
    feature_set = "pst_slotted"
    hidden = (64,)
    architecture = "Sparse network: 768 slotted piece-square inputs, 64-1 (2 layers)"


class NnW64x16Model(NeuralEvalModel):
    """768 slotted piece-square inputs, a 64 accumulator, 16 and one output (3 layers)."""

    name = "net_pst_w64x16"
    feature_set = "pst_slotted"
    hidden = (64, 16)
    architecture = "Sparse network: 768 slotted piece-square inputs, 64-16-1 (3 layers)"


class NnHybridW32Model(NeuralEvalModel):
    """768 piece-square plus 22 term inputs, a 32 accumulator, one output."""

    name = "net_hybrid_w32"
    feature_set = "pst_hybrid"
    hidden = (32,)
    architecture = "Sparse network: 768 piece-square plus 22 term inputs, 32-1 (2 layers)"


class NnHybridW64Model(NeuralEvalModel):
    """768 piece-square plus 22 term inputs, a 64 accumulator, one output."""

    name = "net_hybrid_w64"
    feature_set = "pst_hybrid"
    hidden = (64,)
    architecture = "Sparse network: 768 piece-square plus 22 term inputs, 64-1 (2 layers)"


class NnHybridW128Model(NeuralEvalModel):
    """768 piece-square plus 22 term inputs, a 128 accumulator, one output."""

    name = "net_hybrid_w128"
    feature_set = "pst_hybrid"
    hidden = (128,)
    architecture = "Sparse network: 768 piece-square plus 22 term inputs, 128-1 (2 layers)"


class NnHybridK4W64Model(NeuralEvalModel):
    """Hybrid inputs in 4 king buckets, a 64 accumulator, one output."""

    name = "net_hybrid_k4_w64"
    feature_set = "pst_hybrid_k4"
    hidden = (64,)
    architecture = (
        "Sparse network: 4 king buckets of 768 piece-square inputs plus 22 term inputs, 64-1"
    )


class NnHybridK8W64Model(NeuralEvalModel):
    """Hybrid inputs in 8 king buckets, a 64 accumulator, one output."""

    name = "net_hybrid_k8_w64"
    feature_set = "pst_hybrid_k8"
    hidden = (64,)
    architecture = (
        "Sparse network: 8 king buckets of 768 piece-square inputs plus 22 term inputs, 64-1"
    )


class NnK4W64Model(NeuralEvalModel):
    """Piece squares in 4 king buckets, no terms, a 64 accumulator, one output."""

    name = "net_pst_k4_w64"
    feature_set = "pst_k4"
    hidden = (64,)
    architecture = "Sparse network: 4 king buckets of 768 piece-square inputs, 64-1 (2 layers)"


class NnW64BothModel(NeuralEvalModel):
    """Slotted inputs seen from both sides, a shared 64 accumulator, one output."""

    name = "net_pst_w64_both"
    feature_set = "pst_slotted"
    hidden = (64,)
    perspective = True
    architecture = (
        "Sparse network: 768 slotted piece-square inputs seen from both sides over one shared "
        "64 accumulator, 128-1 (2 layers)"
    )


class NnThreatsW64BothModel(NeuralEvalModel):
    """Slotted inputs plus five threat inputs, seen from both sides, a shared 64 accumulator."""

    name = "net_pst_threats_w64_both"
    feature_set = "pst_threats"
    hidden = (64,)
    perspective = True
    architecture = (
        "Sparse network: 768 slotted piece-square inputs seen from both sides plus 5 threat "
        "inputs of the mover, one shared 64 accumulator, 128-1 (2 layers)"
    )


class NnK4W64BothModel(NeuralEvalModel):
    """Piece squares in 4 king buckets seen from both sides, a shared 64 accumulator."""

    name = "net_pst_k4_w64_both"
    feature_set = "pst_k4"
    hidden = (64,)
    perspective = True
    architecture = (
        "Sparse network: 4 king buckets of 768 piece-square inputs seen from both sides, each "
        "side in the bucket of its own king, one shared 64 accumulator, 128-1 (2 layers)"
    )


class NnK4ThreatsW64BothModel(NeuralEvalModel):
    """King buckets and threat inputs, seen from both sides, a shared 64 accumulator."""

    name = "net_pst_k4_threats_w64_both"
    feature_set = "pst_k4_threats"
    hidden = (64,)
    perspective = True
    architecture = (
        "Sparse network: 4 king buckets of 768 piece-square inputs seen from both sides plus 5 "
        "threat inputs of the mover, one shared 64 accumulator, 128-1 (2 layers)"
    )


#: The models of the project by name.
MODELS: Dict[str, Type[NeuralEvalModel]] = {
    cls.name: cls
    for cls in (
        NnW32Model,
        NnW64Model,
        NnW64x16Model,
        NnW64BothModel,
        NnThreatsW64BothModel,
        NnK4W64BothModel,
        NnK4ThreatsW64BothModel,
        NnHybridW32Model,
        NnHybridW64Model,
        NnHybridW128Model,
        NnHybridK4W64Model,
        NnHybridK8W64Model,
        NnK4W64Model,
    )
}


def available_models() -> List[str]:
    """Names of the models the laboratory defines.

    :return: Sorted list of names.
    """
    return sorted(MODELS)


def get_model_class(name: str) -> Type[NeuralEvalModel]:
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


def load_model(spec: str, path: Optional[Path] = None) -> NeuralEvalModel:
    """Load the saved model named by ``spec``.

    :param spec: Model name, or ``name@tag`` for a tagged artifact.
    :param path: Saved file; defaults to :func:`chess_lab.models.store.loader.artifact_path`.
    :return: The loaded model.
    :raises KeyError: If the name is unknown.
    :raises FileNotFoundError: If the model file does not exist (train it first).
    """
    name, _ = split_spec(spec)
    return get_model_class(name).load(path or artifact_path(spec))


def load_evaluator(spec: str, path: Optional[Path] = None) -> "_core.Evaluator":
    """The core evaluation of the saved model named by ``spec``.

    :param spec: Model name, or ``name@tag`` for a tagged artifact.
    :param path: Saved file; defaults to :func:`chess_lab.models.store.loader.artifact_path`.
    :return: Evaluator that runs inside the core.
    :raises KeyError: If the name is unknown.
    :raises FileNotFoundError: If the model file does not exist (train it first).
    :raises ValueError: If the weights do not fit the feature set.
    """
    return load_model(spec, path).evaluator()
