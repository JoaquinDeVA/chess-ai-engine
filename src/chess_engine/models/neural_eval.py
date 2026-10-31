"""Sparse-input network as the app sees it: layers read from a file and evaluated by the core.

The first weight layer is an accumulator over the slotted features; the core runs the network
(:mod:`chess_engine.models.native`). Initializing and training it (Keras) is the job of
the laboratory.
"""

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .linear_eval import LinearEvalModel, _feature_set_of
from .registry import register_model

Layer = Tuple[np.ndarray, np.ndarray]


class NeuralEvalModel(LinearEvalModel):
    """Feed-forward network over sparse inputs, with weights in NumPy arrays.

    :param layers: ``(weights, bias)`` per weight layer, with shapes ``(inputs, hidden[0])``,
        ``(hidden[0], hidden[1])``, ..., ``(hidden[-1], 1)``.
    :param rep: Representation of the training target as a plain dictionary.
    :cvar hidden: Sizes of the hidden layers, the first being the accumulator.
    """

    hidden: ClassVar[Tuple[int, ...]] = ()

    def __init__(self, layers: Sequence[Layer], rep: Optional[Dict[str, Any]] = None):
        self.rep = dict(rep) if rep else {}
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
        :raises TypeError: If called on the abstract base, which has no feature set.
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If the file was saved with another feature set or hidden sizes, or
            its layers do not chain.
        """
        feature_set = _feature_set_of(cls)
        path = Path(path) if path else cls.default_path()
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            if meta["features"] != feature_set:
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
        meta = {
            "features": _feature_set_of(type(self)),
            "rep": self.rep,
            "hidden": list(type(self).hidden),
        }
        # A file object keeps numpy from appending ".npz" to the requested name.
        with open(path, "wb") as handle:
            np.savez(handle, meta=np.array(json.dumps(meta)), **arrays)
        return path


@register_model("net4_pst")
class Nn4Model(NeuralEvalModel):
    """768 slotted piece-square inputs, a 256 accumulator, 32, 32 and one output (4 layers)."""

    feature_set = "pst_slotted"
    hidden = (256, 32, 32)
    architecture = "Sparse network: 768 slotted piece-square inputs, 256-32-32-1 (4 layers)"
