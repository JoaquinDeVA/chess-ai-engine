"""Training of the sparse network: initial weights and the Keras fit.

The first weight layer is the accumulator: the rows of the active features are summed, and the
sum is clipped to ``[0, 1]``. The remaining hidden layers use ReLU and the output is linear.
Keras (imported lazily by :func:`fit_stream`) trains the network and its weights are exported
to NumPy arrays, which are what the saved model stores.
"""

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from chess_lab.models.definitions import Layer, NeuralEvalModel
from chess_lab.models.features import features_of
from chess_lab.models.target import Representation

from .epochs import (
    DEFAULT_FIT_CONFIG,
    EncodedFrame,
    encode_frame,
    encoded_frames,
    losses,
    run_epochs,
)

#: Standard deviation of the initial accumulator weights (He-normal is used for dense layers).
_ACCUMULATOR_STD = 0.05
#: Seed of the initial weights, so that an untrained model is reproducible.
_INIT_SEED = 0
#: Intra-op threads of the training process; the other cores feed the featurization pool.
_INTRA_OP_THREADS = 2
#: Inputs of one side of the slotted features, and the offset between the two sides' blocks.
_INPUTS_PER_SIDE = 768
_OPPONENT_BLOCK = 384
#: First row of the rival's king inside a bucket (rival block, 5 piece types of 64 before it).
_RIVAL_KING_ROWS = _OPPONENT_BLOCK + 5 * 64
#: Rows per call of the validation forward pass.
_PREDICT_BATCH = 4096


def new_model(model_class: type, rep: Optional[Representation] = None) -> NeuralEvalModel:
    """An untrained network: random He-normal weights, zero biases.

    :param model_class: A concrete :class:`NeuralEvalModel` subclass.
    :param rep: Representation of the target; defaults to :class:`Representation`.
    :return: The model, with ``rep`` saved as a dictionary.
    """
    rep = rep if rep is not None else Representation()
    return model_class(layers=_initial_layers(model_class.layer_shapes()), rep=rep.to_dict())


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


def _import_keras() -> Any:
    """Import Keras lazily: only training needs it."""
    try:
        import keras
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError("training needs the 'ml' extra: pip install -e '.[ml]'") from error
    return keras


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

        def __init__(
            self,
            rows: int,
            units: int,
            l2: float,
            both: bool = False,
            buckets: int = 1,
            **kwargs: Any,
        ):
            super().__init__(**kwargs)
            self.rows, self.units, self.l2, self.both = rows, units, l2, both
            self.buckets = buckets

        def build(self, _input_shape: Any) -> None:
            # L2(c) penalises c * sum(w**2); half the coefficient gives gradient l2 * w.
            self.table = self.add_weight(
                name="table",
                shape=(self.rows, self.units),
                initializer="zeros",
                regularizer=keras.regularizers.L2(0.5 * self.l2),
            )
            self.bias = self.add_weight(name="bias", shape=(self.units,), initializer="zeros")

        def _sum(self, index: Any, value: Any) -> Any:
            rows = keras.ops.take(self.table, index, axis=0)
            total = keras.ops.sum(rows * keras.ops.expand_dims(value, -1), axis=1)
            return keras.ops.clip(total + self.bias, 0.0, 1.0)

        def call(self, inputs: Any) -> Any:
            index, value = inputs
            mover = self._sum(index, value)
            if not self.both:
                return mover
            # The rival's view of a piece is the other block of the same square, in the bucket of
            # the rival's king; the dense entries (past the pieces) have no rival view. Feature
            # ``i`` is index ``i + 1`` and index 0 is padding.
            feature = index - 1
            piece = keras.ops.logical_and(index > 0, feature < self.buckets * _INPUTS_PER_SIDE)
            within = keras.ops.mod(feature, _INPUTS_PER_SIDE)
            swapped = keras.ops.mod(within + _OPPONENT_BLOCK, _INPUTS_PER_SIDE)
            shift = 0
            if self.buckets > 1:
                # The rival king is the entry of the rival block's king rows (704 to 767), whose
                # square is already seen from the rival's home rank.
                is_king = keras.ops.logical_and(piece, within >= _RIVAL_KING_ROWS)
                king = keras.ops.sum(keras.ops.where(is_king, within - _RIVAL_KING_ROWS, 0), axis=1)
                width = 8 // (self.buckets // 2)
                bucket = 2 * (keras.ops.mod(king, 8) // width) + keras.ops.cast(
                    king // 8 >= 2, "int32"
                )
                shift = keras.ops.expand_dims(bucket * _INPUTS_PER_SIDE, -1)
            rival_index = keras.ops.where(piece, shift + swapped + 1, 0)
            rival = self._sum(rival_index, keras.ops.where(piece, value, 0.0))
            return keras.ops.concatenate([mover, rival], axis=-1)

    return SparseAccumulator


def _keras_network(
    keras: Any,
    shapes: Sequence[Tuple[int, int]],
    width: int,
    l2: float,
    both: bool = False,
    buckets: int = 1,
) -> Tuple[Any, List[Any]]:
    """The training network, with the layers of ``shapes``.

    :param keras: The ``keras`` module.
    :param shapes: ``(fan_in, fan_out)`` of every weight layer, accumulator first.
    :param width: Slots per input row.
    :param l2: Weight decay of the accumulator.
    :param both: Whether the second layer reads the accumulators of both sides, one table.
    :param buckets: King buckets of the piece squares (the rival's view needs them).
    :return: The model (inputs: indices, values; output: logit) and its weight layers in order.
    """
    accumulator = _accumulator_class(keras)(
        rows=shapes[0][0] + 1,
        units=shapes[0][1],
        l2=l2,
        both=both,
        buckets=buckets,
        name="accumulator",
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


def fit_stream(
    model: NeuralEvalModel,
    batches: Iterable[pd.DataFrame],
    validation: pd.DataFrame,
    config: Dict[str, Any],
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train with Keras (Adam, binary cross-entropy against soft targets).

    Training starts from the current layers. ``batches`` is read once per epoch, so it must be
    re-iterable. The layers of the epoch with the lowest validation loss are kept. The train
    losses include the ``l2`` penalty of the accumulator (negligible at the default ``l2``).

    With ``workers > 1`` the frames are featurized in a process pool, in order, while the
    main process trains; the result is the same as with one worker.

    :param model: The network to train; its layers are replaced by the best epoch.
    :param batches: DataFrames with columns ``fen, cp, mate, white_to_move``.
    :param validation: Same columns, used for early stopping; must not be empty.
    :param config: Keys ``l2``, ``lr``, ``batch_size``, ``max_epochs``, ``patience``,
        ``workers``; missing keys take :data:`DEFAULT_FIT_CONFIG`.
    :param cache_dir: Directory of the encoded frames, or None for no cache.
    :return: Summary with ``epochs``, ``best_epoch``, ``train_losses``, ``val_losses``,
        ``train_rows``, ``seconds`` and the ``config`` used.
    :raises ValueError: If an epoch yields no rows, the validation set is empty, or a
        validation loss is not finite.
    """
    cfg = {**DEFAULT_FIT_CONFIG, **config}
    started = time.perf_counter()
    if len(validation) == 0:
        raise ValueError("validation set is empty; early stopping needs validation rows")
    features, rep = features_of(model), Representation.from_dict(model.rep)
    keras = _training_keras()
    width = features.max_entries()
    batch_size = int(cfg["batch_size"])
    network, parts = _keras_network(
        keras,
        [w.shape for w, _ in model.layers],
        width,
        float(cfg["l2"]),
        model.perspective,
        getattr(features, "buckets", 1),
    )
    _assign(parts, model.layers)
    network.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(cfg["lr"])),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    val_index, val_value, val_y = _padded(encode_frame(features, rep, validation), width)

    def train_batch(padded: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[float, int]:
        index, value, targets = padded
        loss_sum = 0.0
        for start in range(0, len(targets), batch_size):
            stop = min(len(targets), start + batch_size)
            loss = network.train_on_batch(
                [index[start:stop], value[start:stop]], targets[start:stop, None]
            )
            loss_sum += float(loss) * (stop - start)
        return loss_sum, len(targets)

    def validation_loss() -> float:
        val_z = network.predict(
            [val_index, val_value], batch_size=_PREDICT_BATCH, verbose=0
        ).reshape(-1)
        return float(np.mean(losses(val_z.astype(np.float64), val_y)))

    summary, best_layers = run_epochs(
        cfg=cfg,
        started=started,
        frames=lambda executor: encoded_frames(
            features, rep, batches, int(cfg["workers"]), executor, cache_dir
        ),
        train_batch=train_batch,
        prepare=lambda encoded: _padded(encoded, width),
        validation_loss=validation_loss,
        get_lr=lambda: float(network.optimizer.learning_rate.numpy()),
        set_lr=network.optimizer.learning_rate.assign,
        snapshot=lambda: _export(parts),
        initial_best=model.layers,
    )
    model.layers = best_layers
    del summary["lr_history"]  # not part of this model's summary
    return summary
