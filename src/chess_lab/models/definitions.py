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


#: Most units of the output layer: one per piece count (the core has the same bound).
MAX_OUTPUT_BUCKETS = 32


def output_bucket(pieces: int, buckets: int) -> int:
    """The output unit of a position, as the core chooses it.

    :param pieces: Pieces on the board, kings included (2 to 32).
    :param buckets: Units of the output layer.
    :return: ``(pieces - 1) * buckets // 32``: with eight buckets, four piece counts each.
    """
    return min((pieces - 1) * buckets // MAX_OUTPUT_BUCKETS, buckets - 1)


class NeuralEvalModel:
    """Feed-forward network over the sparse inputs of ``feature_set``, saved in one ``.npz`` file.

    Two weight layers: the accumulator over the sparse inputs, and an output layer over the
    accumulators of both sides, whose value is the logit of the expected score of the player to
    move. With several output buckets, the position reads one unit of the output layer, chosen
    by its number of pieces (see :func:`output_bucket`).

    :param layers: ``(weights, bias)`` per weight layer, with shapes ``(inputs, width)`` and
        ``(2 * width, output_buckets)``.
    :param rep: Representation of the training target as a plain dictionary, saved untouched.
    :cvar name: Model name, also used for the artifact and card file names.
    :cvar feature_set: Name of the feature set, one of ``chess_lab.models.features.FEATURE_SETS``.
    :cvar architecture: One-line description of the architecture, for the model card.
    :cvar hidden: ``(width,)``: the width of the accumulator, a multiple of 8 (the core reads it
        eight values at a time); there is no other hidden layer.
    :cvar families: Which families of the ``pst_ctx`` inputs the network reads, among
        :data:`CTX_FAMILIES`; empty for the other feature sets.
    :cvar output_buckets: Units of the output layer, from 1 to :data:`MAX_OUTPUT_BUCKETS`.
    """

    name: ClassVar[str]
    feature_set: ClassVar[str]
    architecture: ClassVar[str] = ""
    hidden: ClassVar[Tuple[int, ...]] = ()
    families: ClassVar[Tuple[str, ...]] = ()
    output_buckets: ClassVar[int] = 1

    @classmethod
    def input_rows(cls) -> int:
        """Rows of the accumulator table: the inputs the network reads.

        With ``pst_ctx`` the king buckets multiply the piece rows, and each of the material and
        pawn-file families adds its rows; the bucket entries themselves have no row.

        :return: Number of rows of the first layer.
        """
        from .features import (
            FEATURE_SETS,
            KING_BASE,
            KING_BUCKETS,
            MATERIAL_BASE,
            PAWN_FILES_BASE,
        )

        if cls.feature_set != "pst_ctx":
            return FEATURE_SETS[cls.feature_set].size()
        pieces = MATERIAL_BASE * (KING_BUCKETS if "king_buckets" in cls.families else 1)
        material = PAWN_FILES_BASE - MATERIAL_BASE if "material" in cls.families else 0
        pawns = KING_BASE - PAWN_FILES_BASE if "pawn_files" in cls.families else 0
        return pieces + material + pawns

    @classmethod
    def layer_shapes(cls) -> List[Tuple[int, int]]:
        """``(fan_in, fan_out)`` of every weight layer, accumulator first.

        :return: The accumulator and the output layer; the output reads the accumulators of both
            sides (the mover's, then the rival's, over the same first-layer weights), so it has
            twice the inputs.
        :raises ValueError: If ``hidden`` is not one width that is a multiple of 8.
        """
        if len(cls.hidden) != 1 or cls.hidden[0] % 8:
            raise ValueError(
                f"{cls.name}: hidden must be one accumulator width, a multiple of 8, "
                f"not {cls.hidden}"
            )
        width = cls.hidden[0]
        return [(cls.input_rows(), width), (2 * width, cls.output_buckets)]

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
        return _core.Evaluator.network(self.feature_set, layers, list(self.families))

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
            if tuple(meta.get("families", ())) != cls.families:
                raise ValueError(f"{path} was saved with families {meta.get('families', ())}")
            if tuple(meta["hidden"]) != cls.hidden:
                raise ValueError(
                    f"{path} was saved with hidden sizes {tuple(meta['hidden'])}, "
                    f"expected {cls.hidden}"
                )
            layers = [(data[f"W{i}"], data[f"b{i}"]) for i in range(len(cls.hidden) + 1)]
        fan_outs = (*cls.hidden, cls.output_buckets)
        for position, ((weights, bias), fan_out) in enumerate(zip(layers, fan_outs, strict=True)):
            if weights.shape[1] != fan_out or bias.shape != (fan_out,):
                raise ValueError(
                    f"{path}: layer {position} has shapes {weights.shape} and {bias.shape}, "
                    f"expected {fan_out} outputs"
                )
        if layers[1][0].shape[0] != 2 * cls.hidden[0]:
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
        meta = {
            "features": self.feature_set,
            "rep": self.rep,
            "hidden": list(self.hidden),
            "families": list(self.families),
        }
        # A file object keeps numpy from appending ".npz" to the requested name.
        with open(path, "wb") as handle:
            np.savez(handle, meta=np.array(json.dumps(meta)), **arrays)
        return path


class NnW64BothModel(NeuralEvalModel):
    """Slotted inputs seen from both sides, a shared 64 accumulator, one output."""

    name = "net_pst_w64_both"
    feature_set = "pst_slotted"
    hidden = (64,)
    architecture = (
        "Sparse network: 768 slotted piece-square inputs seen from both sides over one shared "
        "64 accumulator, 128-1 (2 layers)"
    )


class NnW128BothModel(NeuralEvalModel):
    """Slotted inputs seen from both sides, a shared 128 accumulator, one output."""

    name = "net_pst_w128_both"
    feature_set = "pst_slotted"
    hidden = (128,)
    architecture = (
        "Sparse network: 768 slotted piece-square inputs seen from both sides over one shared "
        "128 accumulator, 256-1 (2 layers)"
    )


class NnW128Ob8BothModel(NeuralEvalModel):
    """As ``net_pst_w128_both``, with one output unit per four piece counts (eight buckets)."""

    name = "net_pst_w128_ob8_both"
    feature_set = "pst_slotted"
    hidden = (128,)
    output_buckets = 8
    architecture = (
        "Sparse network: 768 slotted piece-square inputs seen from both sides over one shared "
        "128 accumulator, 256-8 with one output unit per 4 piece counts (2 layers)"
    )


class NnW256BothModel(NeuralEvalModel):
    """Slotted inputs seen from both sides, a shared 256 accumulator, one output."""

    name = "net_pst_w256_both"
    feature_set = "pst_slotted"
    hidden = (256,)
    architecture = (
        "Sparse network: 768 slotted piece-square inputs seen from both sides over one shared "
        "256 accumulator, 512-1 (2 layers)"
    )


class NnThreatsW64BothModel(NeuralEvalModel):
    """Slotted inputs plus five threat inputs, seen from both sides, a shared 64 accumulator."""

    name = "net_pst_threats_w64_both"
    feature_set = "pst_threats"
    hidden = (64,)
    architecture = (
        "Sparse network: 768 slotted piece-square inputs seen from both sides plus 5 threat "
        "inputs of the mover, one shared 64 accumulator, 128-1 (2 layers)"
    )


#: Families of the ``pst_ctx`` inputs by the letter of the model names.
CTX_FAMILIES = {"m": "material", "p": "pawn_files", "k": "king_buckets"}


def _ctx_model(letters: str, width: int) -> Type[NeuralEvalModel]:
    """A ``pst_ctx`` network over the families named by ``letters``.

    :param letters: Some of ``m`` (material), ``p`` (pawn files) and ``k`` (own king buckets).
    :param width: Accumulator width.
    :return: The model class ``net_ctx_<letters>_w<width>_both``.
    """
    families = tuple(CTX_FAMILIES[letter] for letter in letters)
    return type(
        f"NnCtx{letters.upper()}W{width}BothModel",
        (NeuralEvalModel,),
        {
            "__doc__": f"Slotted inputs plus {', '.join(families)}, both sides, {width} accumulator.",
            "name": f"net_ctx_{letters}_w{width}_both",
            "feature_set": "pst_ctx",
            "hidden": (width,),
            "families": families,
            "architecture": (
                f"Sparse network: 768 slotted piece-square inputs plus {', '.join(families)} "
                f"seen from both sides over one shared {width} accumulator, {2 * width}-1 "
                "(2 layers)"
            ),
        },
    )


#: The models of the project by name.
MODELS: Dict[str, Type[NeuralEvalModel]] = {
    cls.name: cls
    for cls in (
        NnW64BothModel,
        NnW128BothModel,
        NnW128Ob8BothModel,
        NnW256BothModel,
        NnThreatsW64BothModel,
        *(_ctx_model(letters, 128) for letters in ("m", "p", "k", "mpk")),
        *(_ctx_model(letters, 256) for letters in ("m", "mpk")),
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
