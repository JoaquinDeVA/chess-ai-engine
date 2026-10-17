"""Sparse-input neural evaluators: an accumulator over slotted features, then dense layers.

The first weight layer is the accumulator: the rows of the active features are summed, and the
sum is clipped to ``[0, 1]``. The remaining hidden layers use ReLU and the output is linear. The
play path is NumPy only; Keras (imported lazily by :meth:`NeuralEvalModel.fit_stream`) trains the
network, and its weights are exported to NumPy arrays.
"""

import json
import math
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple

import chess
import numpy as np
import pandas as pd

from chess_engine.training.representation import Representation

from .keras_model import _import_keras
from .linear_eval import (
    DEFAULT_FIT_CONFIG,
    EncodedFrame,
    LinearEvalModel,
    _features_of,
    _featurization_pool,
    _losses,
)
from .linear_features import SlottedPstFeatures
from .registry import register_model

Layer = Tuple[np.ndarray, np.ndarray]

#: Standard deviation of the initial accumulator weights (He-normal is used for dense layers).
_ACCUMULATOR_STD = 0.05
#: Seed of the initial weights, so that an untrained model is reproducible.
_INIT_SEED = 0
#: Intra-op threads of the training process; the other cores feed the featurization pool.
_INTRA_OP_THREADS = 2
#: Rows per call of the validation forward pass.
_PREDICT_BATCH = 4096


def _initial_layers(shapes: Sequence[Tuple[int, int]]) -> List[Layer]:
    """Random weights for the given layer shapes, zero biases.

    :param shapes: ``(fan_in, fan_out)`` of every weight layer, accumulator first.
    :return: One ``(weights, bias)`` pair per layer.
    """
    rng = np.random.default_rng(_INIT_SEED)
    layers: List[Layer] = []
    for position, (fan_in, fan_out) in enumerate(shapes):
        std = _ACCUMULATOR_STD if position == 0 else math.sqrt(2.0 / fan_in)
        layers.append((rng.normal(0.0, std, size=(fan_in, fan_out)), np.zeros(fan_out)))
    return layers


def _padded(encoded: EncodedFrame, width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn sparse entries into fixed-width arrays for the accumulator.

    Feature ``i`` becomes index ``i + 1``; index ``0`` is padding, whose value is ``0``.

    :param encoded: ``(rows, cols, values, targets)`` as made by :func:`encode_frame`; ``rows``
        is sorted.
    :param width: Number of slots per row; at least the most entries of any row.
    :return: ``(index, value, targets)`` with ``index`` of dtype int32 and ``value`` of float32,
        both of shape ``(rows, width)``.
    """
    rows, cols, vals, targets = encoded
    count = len(targets)
    starts = np.searchsorted(rows, np.arange(count))
    position = np.arange(rows.size) - starts[rows]
    index = np.zeros((count, width), dtype=np.int32)
    value = np.zeros((count, width), dtype=np.float32)
    index[rows, position] = cols + 1
    value[rows, position] = vals
    return index, value, targets


def _training_keras() -> Any:
    """Keras, with the training process limited to a few intra-op threads.

    :return: The ``keras`` module.
    """
    keras = _import_keras()
    import tensorflow as tf

    try:
        tf.config.threading.set_intra_op_parallelism_threads(_INTRA_OP_THREADS)
    except RuntimeError:  # the TensorFlow runtime is already running: keep its setting
        pass
    return keras


def _accumulator_class(keras: Any) -> type:
    """The accumulator layer class, defined once Keras is imported.

    :param keras: The ``keras`` module.
    :return: A Keras layer class; index ``0`` is padding and carries value ``0``.
    """

    class SparseAccumulator(keras.layers.Layer):
        """Value-weighted sum of the rows of a table, clipped to ``[0, 1]``."""

        def __init__(self, rows: int, units: int, l2: float, **kwargs: Any):
            super().__init__(**kwargs)
            self.rows, self.units, self.l2 = rows, units, l2

        def build(self, input_shape: Any) -> None:
            # L2(c) penalises c * sum(w**2); half the coefficient gives gradient l2 * w.
            self.table = self.add_weight(
                name="table",
                shape=(self.rows, self.units),
                initializer="zeros",
                regularizer=keras.regularizers.L2(0.5 * self.l2),
            )
            self.bias = self.add_weight(name="bias", shape=(self.units,), initializer="zeros")

        def call(self, inputs: Any) -> Any:
            index, value = inputs
            rows = keras.ops.take(self.table, index, axis=0)
            total = keras.ops.sum(rows * keras.ops.expand_dims(value, -1), axis=1)
            return keras.ops.clip(total + self.bias, 0.0, 1.0)

    return SparseAccumulator


def _keras_network(
    keras: Any, shapes: Sequence[Tuple[int, int]], width: int, l2: float
) -> Tuple[Any, List[Any]]:
    """The training network, with the layers of ``shapes``.

    :param keras: The ``keras`` module.
    :param shapes: ``(fan_in, fan_out)`` of every weight layer, accumulator first.
    :param width: Slots per input row.
    :param l2: Weight decay of the accumulator.
    :return: The model (inputs: indices, values; output: logit) and its weight layers in order.
    """
    accumulator = _accumulator_class(keras)(
        rows=shapes[0][0] + 1, units=shapes[0][1], l2=l2, name="accumulator"
    )
    index_in = keras.Input(shape=(width,), dtype="int32")
    value_in = keras.Input(shape=(width,), dtype="float32")
    x = accumulator([index_in, value_in])
    dense_layers: List[Any] = []
    last = len(shapes) - 1
    for position, (_, units) in enumerate(shapes[1:], start=1):
        activation = "relu" if position < last else None
        layer = keras.layers.Dense(
            units, activation=activation, kernel_initializer="he_normal", name=f"dense{position}"
        )
        x = layer(x)
        dense_layers.append(layer)
    model = keras.Model([index_in, value_in], x)
    return model, [accumulator, *dense_layers]


def _assign(parts: Sequence[Any], layers: Sequence[Layer]) -> None:
    """Copy NumPy weights into the Keras weight layers (accumulator table without padding row).

    :param parts: Accumulator first, then the dense layers.
    :param layers: ``(weights, bias)`` per layer, same order.
    """
    accumulator, dense = parts[0], parts[1:]
    weights, bias = layers[0]
    table = np.vstack([np.zeros((1, weights.shape[1])), weights])
    accumulator.table.assign(table.astype(np.float32))
    accumulator.bias.assign(bias.astype(np.float32))
    for layer, (weights, bias) in zip(dense, layers[1:], strict=True):
        layer.kernel.assign(weights.astype(np.float32))
        layer.bias.assign(bias.astype(np.float32))


def _export(parts: Sequence[Any]) -> List[Layer]:
    """NumPy copies of the Keras weights, accumulator first (padding row dropped).

    :param parts: Accumulator first, then the dense layers.
    :return: ``(weights, bias)`` per layer, in float64.
    """
    accumulator, dense = parts[0], parts[1:]
    layers = [(np.asarray(accumulator.table)[1:], np.asarray(accumulator.bias))]
    layers += [(np.asarray(layer.kernel), np.asarray(layer.bias)) for layer in dense]
    return [(np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64)) for w, b in layers]


class NeuralEvalModel(LinearEvalModel):
    """Feed-forward network over sparse inputs, with weights in NumPy arrays.

    The features must provide ``max_entries(perspective)``, the most non-zero entries of a
    position. Subclasses set ``features`` and ``hidden``; ``hidden[0]`` is the accumulator width.

    :param layers: ``(weights, bias)`` per weight layer, with shapes ``(features.size, hidden[0])``,
        ``(hidden[0], hidden[1])``, ..., ``(hidden[-1], 1)``; random He-normal weights if omitted.
    :param rep: Representation of the target; defaults to :class:`Representation`.
    :cvar features: Sparse encoding of positions.
    :cvar hidden: Sizes of the hidden layers, the first being the accumulator.
    """

    hidden: ClassVar[Tuple[int, ...]] = ()

    def __init__(
        self, layers: Optional[Sequence[Layer]] = None, rep: Optional[Representation] = None
    ):
        self.rep = rep if rep is not None else Representation()
        shapes = self._layer_shapes(self.rep)
        if layers is None:
            layers = _initial_layers(shapes)
        self.layers: List[Layer] = [
            (np.asarray(weights, dtype=np.float64), np.asarray(bias, dtype=np.float64))
            for weights, bias in layers
        ]

    @classmethod
    def _layer_shapes(cls, rep: Representation) -> List[Tuple[int, int]]:
        """``(fan_in, fan_out)`` of every weight layer.

        :param rep: Representation, whose perspective sets the input size.
        :return: Shapes, accumulator first, output last.
        """
        sizes = (_features_of(cls).size(rep.perspective), *cls.hidden, 1)
        return list(zip(sizes[:-1], sizes[1:], strict=True))

    def logit(self, board: chess.Board) -> float:
        """Network output before the logistic function, for the side to move or for White.

        :param board: The position.
        :return: The output of the last layer, in NumPy.
        """
        indices, values = self.features.active(board, self.rep.perspective)
        weights, bias = self.layers[0]
        hidden = np.clip(np.asarray(values) @ weights[indices] + bias, 0.0, 1.0)
        for weights, bias in self.layers[1:-1]:
            hidden = np.maximum(hidden @ weights + bias, 0.0)
        weights, bias = self.layers[-1]
        return float(hidden @ weights[:, 0] + bias[0])

    def count_parameters(self) -> int:
        """Weights and biases of all layers.

        :return: Total number of parameters.
        """
        return sum(weights.size + bias.size for weights, bias in self.layers)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "NeuralEvalModel":
        """Load a saved model.

        :param path: File to load; defaults to :meth:`default_path`.
        :return: The model, with the representation it was trained with.
        :raises TypeError: If called on the abstract base, which has no features.
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If the file was saved with other features or hidden sizes, or its
            layer shapes do not match.
        """
        features = _features_of(cls)
        path = Path(path) if path else cls.default_path()
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            if meta["features"] != features.name:
                raise ValueError(f"{path} was saved with features {meta['features']!r}")
            if tuple(meta["hidden"]) != cls.hidden:
                raise ValueError(
                    f"{path} was saved with hidden sizes {tuple(meta['hidden'])}, "
                    f"expected {cls.hidden}"
                )
            rep = Representation.from_dict(meta["rep"])
            shapes = cls._layer_shapes(rep)
            layers = [(data[f"W{i}"], data[f"b{i}"]) for i in range(len(shapes))]
        for position, ((fan_in, fan_out), (weights, bias)) in enumerate(
            zip(shapes, layers, strict=True)
        ):
            if weights.shape != (fan_in, fan_out) or bias.shape != (fan_out,):
                raise ValueError(
                    f"{path}: layer {position} has shapes {weights.shape} and {bias.shape}, "
                    f"expected {(fan_in, fan_out)} and {(fan_out,)}"
                )
        return cls(layers=layers, rep=rep)

    def save(self, path: Optional[Path] = None) -> Path:
        """Save the layers, the features name, the hidden sizes and the representation.

        :param path: Destination; defaults to :meth:`default_path`.
        :return: The path that was written.
        """
        features = _features_of(type(self))
        path = Path(path) if path else self.default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        for position, (weights, bias) in enumerate(self.layers):
            arrays[f"W{position}"], arrays[f"b{position}"] = weights, bias
        meta = {
            "features": features.name,
            "rep": self.rep.to_dict(),
            "hidden": list(type(self).hidden),
        }
        # A file object keeps numpy from appending ".npz" to the requested name.
        with open(path, "wb") as handle:
            np.savez(handle, meta=np.array(json.dumps(meta)), **arrays)
        return path

    def fit_stream(
        self,
        batches: Iterable[pd.DataFrame],
        validation: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Train with Keras (Adam, binary cross-entropy against soft targets).

        Training starts from the current layers. ``batches`` is read once per epoch, so it must be
        re-iterable. The layers of the epoch with the lowest validation loss are kept. The train
        losses include the ``l2`` penalty of the accumulator (negligible at the default ``l2``).

        With ``workers > 1`` frames are featurized in a process pool, as in
        :meth:`LinearEvalModel.fit_stream`.

        :param batches: DataFrames with columns ``fen, cp, mate, white_to_move``.
        :param validation: Same columns, used for early stopping; must not be empty.
        :param config: Keys ``l2``, ``lr``, ``batch_size``, ``max_epochs``, ``patience``,
            ``workers``; missing keys take :data:`DEFAULT_FIT_CONFIG`.
        :return: Summary with ``epochs``, ``best_epoch``, ``train_losses``, ``val_losses``,
            ``train_rows``, ``seconds`` and the ``config`` used.
        :raises ValueError: If an epoch yields no rows, the validation set is empty, or a
            validation loss is not finite.
        """
        cfg = {**DEFAULT_FIT_CONFIG, **config}
        started = time.perf_counter()
        if len(validation) == 0:
            raise ValueError("validation set is empty; early stopping needs validation rows")
        keras = _training_keras()
        width = self.features.max_entries(self.rep.perspective)
        batch_size = int(cfg["batch_size"])
        workers = int(cfg["workers"])
        model, parts = _keras_network(
            keras, [w.shape for w, _ in self.layers], width, float(cfg["l2"])
        )
        _assign(parts, self.layers)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=float(cfg["lr"])),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        val_index, val_value, val_y = _padded(self._encode(validation), width)

        best_layers, best_loss, best_epoch = self.layers, math.inf, 0
        train_losses: List[float] = []
        val_losses: List[float] = []
        bad_epochs = 0
        train_rows = 0

        with _featurization_pool(workers) as executor:
            for epoch in range(1, int(cfg["max_epochs"]) + 1):
                loss_sum, rows_seen = 0.0, 0
                for encoded in self._encoded_frames(batches, workers, executor):
                    index, value, targets = _padded(encoded, width)
                    for start in range(0, len(targets), batch_size):
                        stop = min(len(targets), start + batch_size)
                        loss = model.train_on_batch(
                            [index[start:stop], value[start:stop]], targets[start:stop, None]
                        )
                        loss_sum += float(loss) * (stop - start)
                        rows_seen += stop - start
                if rows_seen == 0:
                    raise ValueError("batches yielded no rows; pass a re-iterable collection")
                train_rows = rows_seen
                train_losses.append(loss_sum / rows_seen)

                val_z = model.predict(
                    [val_index, val_value], batch_size=_PREDICT_BATCH, verbose=0
                ).reshape(-1)
                val_loss = float(np.mean(_losses(val_z.astype(np.float64), val_y)))
                if not math.isfinite(val_loss):
                    raise ValueError(f"validation loss is not finite at epoch {epoch}: {val_loss}")
                val_losses.append(val_loss)
                if val_loss < best_loss:
                    best_loss, best_epoch, best_layers = val_loss, epoch, _export(parts)
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs > int(cfg["patience"]):
                        break

        self.layers = best_layers
        return {
            "epochs": len(train_losses),
            "best_epoch": best_epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_rows": train_rows,
            "seconds": round(time.perf_counter() - started, 3),
            "config": cfg,
        }


@register_model("nn4")
class Nn4Model(NeuralEvalModel):
    """768 slotted piece-square inputs, a 256 accumulator, 32, 32 and one output (4 layers)."""

    features = SlottedPstFeatures()
    hidden = (256, 32, 32)
    architecture = "Sparse network: 768 slotted piece-square inputs, 256-32-32-1 (4 layers)"
