"""Training of the sparse network: initial weights and the Keras fit.

The first weight layer is the accumulator: the rows of the active features are summed, and the
sum is clipped to ``[0, 1]``. The output is one linear unit over the accumulators of both sides.
Keras (imported lazily by :func:`fit_stream`) trains the network and its weights are exported
to NumPy arrays, which are what the saved model stores.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from chess_lab.models.definitions import Layer, NeuralEvalModel
from chess_lab.models.features import (
    KING_BASE,
    KING_BUCKETS,
    MATERIAL_BASE,
    MATERIAL_PER_SIDE,
    PAWN_FILES_BASE,
    PAWN_FILES_PER_SIDE,
    features_of,
)
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
            families: Sequence[str] = (),
            ctx: bool = False,
            **kwargs: Any,
        ):
            super().__init__(**kwargs)
            self.rows, self.units, self.l2 = rows, units, l2
            self.families, self.ctx = tuple(families), ctx

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

        def _ctx_rows(self, index: Any, value: Any, bucket: Any, swap: bool) -> Tuple[Any, Any]:
            """Table rows and values of one side's view of ``pst_ctx`` entries.

            Piece entries read ``bucket * 768 + piece`` with king buckets, else the piece; the
            material and pawn-file entries read their own rows after the pieces, or nothing when
            their family is off; the bucket entries have no row. ``swap`` gives the rival's view:
            the two sides' blocks of every family exchange places.

            :param index: Feature ``i`` as ``i + 1``, ``0`` for padding.
            :param value: Values of the entries.
            :param bucket: King bucket of the side the view belongs to, shape ``(rows,)``.
            :param swap: Whether this is the rival's view.
            :return: ``(index, value)`` for :meth:`_sum`, with ``0`` where an entry is dropped.
            """
            ops = keras.ops
            feature = index - 1
            valid = index > 0
            is_piece = ops.logical_and(valid, feature < MATERIAL_BASE)
            is_material = ops.logical_and(
                ops.logical_and(valid, feature >= MATERIAL_BASE), feature < PAWN_FILES_BASE
            )
            is_pawn = ops.logical_and(
                ops.logical_and(valid, feature >= PAWN_FILES_BASE), feature < KING_BASE
            )
            piece, material, pawn = feature, feature - MATERIAL_BASE, feature - PAWN_FILES_BASE
            if swap:
                piece = ops.mod(piece + MATERIAL_BASE // 2, MATERIAL_BASE)
                material = ops.mod(material + MATERIAL_PER_SIDE, 2 * MATERIAL_PER_SIDE)
                pawn = ops.mod(pawn + PAWN_FILES_PER_SIDE, 2 * PAWN_FILES_PER_SIDE)
            king = "king_buckets" in self.families
            if king:
                piece = piece + ops.expand_dims(bucket, -1) * MATERIAL_BASE
            piece_rows = MATERIAL_BASE * (KING_BUCKETS if king else 1)
            material_rows = 2 * MATERIAL_PER_SIDE if "material" in self.families else 0
            row = ops.where(is_piece, piece, -1)
            if "material" in self.families:
                row = ops.where(is_material, piece_rows + material, row)
            if "pawn_files" in self.families:
                row = ops.where(is_pawn, piece_rows + material_rows + pawn, row)
            kept = row >= 0
            return ops.where(kept, row + 1, 0), ops.where(kept, value, 0.0)

        def _ctx_call(self, index: Any, value: Any) -> Any:
            ops = keras.ops
            feature = index - 1
            in_buckets = ops.logical_and(index > 0, feature >= KING_BASE)
            own = ops.logical_and(in_buckets, feature < KING_BASE + KING_BUCKETS)
            rival = ops.logical_and(in_buckets, feature >= KING_BASE + KING_BUCKETS)
            own_bucket = ops.sum(ops.where(own, feature - KING_BASE, 0), axis=1)
            rival_bucket = ops.sum(ops.where(rival, feature - KING_BASE - KING_BUCKETS, 0), axis=1)
            mover = self._sum(*self._ctx_rows(index, value, own_bucket, False))
            rival_view = self._sum(*self._ctx_rows(index, value, rival_bucket, True))
            return ops.concatenate([mover, rival_view], axis=-1)

        def call(self, inputs: Any) -> Any:
            index, value = inputs
            if self.ctx:
                return self._ctx_call(index, value)
            mover = self._sum(index, value)
            # The rival's view of a piece is the other block of the same square; the dense entries
            # (past the pieces) have no rival view. Feature ``i`` is index ``i + 1`` and index 0
            # is padding.
            feature = index - 1
            piece = keras.ops.logical_and(index > 0, feature < _INPUTS_PER_SIDE)
            swapped = keras.ops.mod(feature + _OPPONENT_BLOCK, _INPUTS_PER_SIDE)
            rival_index = keras.ops.where(piece, swapped + 1, 0)
            rival = self._sum(rival_index, keras.ops.where(piece, value, 0.0))
            return keras.ops.concatenate([mover, rival], axis=-1)

    return SparseAccumulator


def _keras_network(
    keras: Any,
    shapes: Sequence[Tuple[int, int]],
    width: int,
    l2: float,
    families: Sequence[str] = (),
    ctx: bool = False,
) -> Tuple[Any, List[Any]]:
    """The training network: the accumulator and the output unit of ``shapes``.

    :param keras: The ``keras`` module.
    :param shapes: ``(fan_in, fan_out)`` of the two weight layers, accumulator first.
    :param width: Slots per input row.
    :param l2: Weight decay of the accumulator.
    :param families: Families of the ``pst_ctx`` inputs the network reads.
    :param ctx: Whether the inputs are those of ``pst_ctx``.
    :return: The model (inputs: indices, values; output: logit) and its weight layers in order.
    """
    accumulator = _accumulator_class(keras)(
        rows=shapes[0][0] + 1,
        units=shapes[0][1],
        l2=l2,
        families=families,
        ctx=ctx,
        name="accumulator",
    )
    index_in = keras.Input(shape=(width,), dtype="int32")
    value_in = keras.Input(shape=(width,), dtype="float32")
    x = accumulator([index_in, value_in])
    output = keras.layers.Dense(1, kernel_initializer="he_normal", name="dense1")
    model = keras.Model([index_in, value_in], output(x))
    return model, [accumulator, output]


def _assign(parts: Sequence[Any], layers: Sequence[Layer]) -> None:
    """Copy NumPy weights into the Keras weight layers (accumulator table without padding row).

    :param parts: The accumulator and the output layer.
    :param layers: ``(weights, bias)`` per layer, same order.
    """
    accumulator, output = parts
    (weights, bias), (output_weights, output_bias) = layers
    table = np.vstack([np.zeros((1, weights.shape[1])), weights])
    accumulator.table.assign(table.astype(np.float32))
    accumulator.bias.assign(bias.astype(np.float32))
    output.kernel.assign(output_weights.astype(np.float32))
    output.bias.assign(output_bias.astype(np.float32))


def _export(parts: Sequence[Any]) -> List[Layer]:
    """NumPy copies of the Keras weights, accumulator first (padding row dropped).

    :param parts: The accumulator and the output layer.
    :return: ``(weights, bias)`` per layer, in float64.
    """
    accumulator, output = parts
    layers = [
        (np.asarray(accumulator.table)[1:], np.asarray(accumulator.bias)),
        (np.asarray(output.kernel), np.asarray(output.bias)),
    ]
    return [(np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64)) for w, b in layers]


def checkpoint_config(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """The fit configuration of the run that left the checkpoint ``path``, to resume it as it was.

    :param path: Checkpoint file of :func:`fit_stream`, or None.
    :return: The configuration, or None if there is no checkpoint there.
    """
    if path is None or not Path(path).exists():
        return None
    with np.load(path) as data:
        return json.loads(str(data["loop"]))["config"]


def _save_checkpoint(path: Path, network: Any, loop: Dict[str, Any], best: List[Layer]) -> None:
    """Write the state of the fit after an epoch.

    The loop, the Keras weights and optimizer, and the best layers so far go in one file, replaced
    at once, so a cut while writing keeps the previous one.
    """
    arrays = {"loop": np.array(json.dumps(loop))}
    for i, weights in enumerate(network.get_weights()):
        arrays[f"weights{i}"] = weights
    for i, variable in enumerate(network.optimizer.variables):
        arrays[f"optimizer{i}"] = np.asarray(variable)
    for i, (weights, bias) in enumerate(best):
        arrays[f"best_weights{i}"], arrays[f"best_bias{i}"] = weights, bias
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(partial, path)


def _load_checkpoint(path: Path, network: Any) -> Tuple[Dict[str, Any], List[Layer]]:
    """Put back into ``network`` the weights and optimizer of :func:`_save_checkpoint`.

    :return: The state of the loop and the best layers so far.
    """
    with np.load(path) as data:
        loop = json.loads(str(data["loop"]))
        network.set_weights([data[f"weights{i}"] for i in range(len(network.get_weights()))])
        network.optimizer.build(network.trainable_variables)
        for i, variable in enumerate(network.optimizer.variables):
            variable.assign(data[f"optimizer{i}"])
        best = [(data[f"best_weights{i}"], data[f"best_bias{i}"]) for i in range(2)]
    return loop, best


def fit_stream(
    model: NeuralEvalModel,
    batches: Iterable[pd.DataFrame],
    validation: pd.DataFrame,
    config: Dict[str, Any],
    cache_dir: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
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
    :param checkpoint: File where every epoch leaves the state to resume from, or None. If it
        exists, the fit goes on after its last epoch (with the same ``config``); the caller
        deletes it once the model is saved.
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
        model.families,
        model.feature_set == "pst_ctx",
    )
    _assign(parts, model.layers)
    network.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(cfg["lr"])),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    val_index, val_value, val_y = _padded(encode_frame(features, rep, validation), width)
    resume, initial_best = None, model.layers
    if checkpoint is not None and Path(checkpoint).exists():
        resume, initial_best = _load_checkpoint(Path(checkpoint), network)

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
        initial_best=initial_best,
        resume=resume,
        save=None
        if checkpoint is None
        else lambda loop, best: _save_checkpoint(Path(checkpoint), network, loop, best),
    )
    model.layers = best_layers
    del summary["lr_history"]  # not part of this model's summary
    return summary
