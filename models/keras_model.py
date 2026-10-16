"""Adapter for Keras networks (stored in the ``.keras`` format)."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import EvaluationModel


def _import_keras():
    """Import Keras lazily so that the rest of the engine works without it installed."""
    try:
        import keras
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError("Keras models need the 'ml' extra: pip install -e '.[ml]'") from error
    return keras


class KerasModel(EvaluationModel):
    """Evaluation model backed by a Keras network with one scalar output.

    Subclasses set ``extractor`` and implement :meth:`build_network`; the network must have
    one input per array produced by the extractor.

    :param network: A Keras model.
    """

    artifact_suffix = ".keras"

    def __init__(self, network: Any):
        self.network = network

    @classmethod
    def build_network(cls, config: Dict[str, Any]) -> Any:
        """Create an untrained network.

        :param config: Hyper-parameters from the training configuration.
        :return: An uncompiled ``keras.Model``.
        """
        raise NotImplementedError(f"{cls.__name__} does not define build_network()")

    @classmethod
    def new(cls, config: Optional[Dict[str, Any]] = None) -> "KerasModel":
        """Create an untrained model.

        :param config: Hyper-parameters passed to :meth:`build_network`.
        :return: A model ready for :meth:`fit`.
        """
        return cls(cls.build_network(config or {}))

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "KerasModel":
        """Load a ``.keras`` file.

        :param path: File to load; defaults to :meth:`default_path`.
        :return: The loaded model.
        :raises FileNotFoundError: If the file does not exist.
        """
        keras = _import_keras()
        path = Path(path) if path else cls.default_path()
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        return cls(keras.models.load_model(path, compile=False))

    def save(self, path: Optional[Path] = None) -> Path:
        """Save the network.

        :param path: Destination; defaults to :meth:`default_path`.
        :return: The path that was written.
        """
        path = Path(path) if path else self.default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.network.save(path)
        return path

    def predict_target(self, inputs: Tuple[np.ndarray, ...]) -> np.ndarray:
        """Run the network on an encoded batch.

        Calling the network directly is much faster than ``Model.predict`` for the tiny batches
        used inside a game tree search.

        :param inputs: Encoded batch, one array per network input.
        :return: Predictions, shape ``(n,)``.
        """
        feed = list(inputs) if len(inputs) > 1 else inputs[0]
        return np.asarray(self.network(feed, training=False)).reshape(-1)

    def fit(
        self,
        inputs: Tuple[np.ndarray, ...],
        targets: np.ndarray,
        validation: Optional[Tuple[Tuple[np.ndarray, ...], np.ndarray]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train with Adam, early stopping and learning-rate decay.

        Recognized ``config`` keys: ``epochs`` (default 50), ``batch_size`` (256),
        ``learning_rate`` (1e-3), ``loss`` (``"mse"`` or ``"huber"``), ``huber_delta`` (0.1),
        ``patience`` (7).

        :param inputs: Encoded training positions.
        :param targets: Normalized targets.
        :param validation: Optional ``(inputs, targets)`` for monitoring and early stopping.
        :param config: Hyper-parameters, see above.
        :return: Summary with epochs run and the final losses.
        """
        keras = _import_keras()
        config = config or {}
        loss = (
            keras.losses.Huber(delta=config.get("huber_delta", 0.1))
            if config.get("loss", "mse") == "huber"
            else "mse"
        )
        self.network.compile(
            optimizer=keras.optimizers.Adam(config.get("learning_rate", 1e-3)),
            loss=loss,
            metrics=["mae"],
        )
        monitor = "val_loss" if validation is not None else "loss"
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=monitor, patience=config.get("patience", 7), restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3, min_lr=1e-5),
        ]
        feed = list(inputs) if len(inputs) > 1 else inputs[0]
        validation_data = None
        if validation is not None:
            val_inputs, val_targets = validation
            val_feed = list(val_inputs) if len(val_inputs) > 1 else val_inputs[0]
            validation_data = (val_feed, val_targets)

        history = self.network.fit(
            feed,
            targets,
            validation_data=validation_data,
            epochs=config.get("epochs", 50),
            batch_size=config.get("batch_size", 256),
            callbacks=callbacks,
            verbose=config.get("verbose", 1),
        ).history
        summary = {
            "epochs": len(history["loss"]),
            "train_samples": int(len(targets)),
            "final_loss": float(history["loss"][-1]),
        }
        if "val_loss" in history:
            summary["final_val_loss"] = float(history["val_loss"][-1])
            summary["final_val_mae"] = float(history["val_mae"][-1])
        return summary

    def count_parameters(self) -> int:
        """Number of parameters of the network.

        :return: Parameter count.
        """
        return int(self.network.count_params())
