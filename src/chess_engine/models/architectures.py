"""Model architectures shipped with the project.

* ``ridge_features``: scikit-learn ``StandardScaler -> Ridge`` on 29 hand-crafted features.
* ``mlp_features``: a tiny dense network on the same features, evaluated with plain NumPy.
* ``cnn_resnet``: residual convolutional network on an ``8x8x12`` piece tensor (+ 9 extras).
* ``cnn_light``: smaller convolutional network with the same inputs (load-only).
"""

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import BitboardFeatures, VectorFeatures
from .keras_model import KerasModel, _import_keras
from .registry import register_model
from .sklearn_model import SklearnModel


@register_model("ridge_features")
class RidgeFeatures(SklearnModel):
    """Ridge regression (alpha 10) on standardized :class:`VectorFeatures`."""

    extractor = VectorFeatures()
    architecture = "StandardScaler -> Ridge(alpha=10) on 29 features"

    @classmethod
    def build_estimator(cls, config: Dict[str, Any]) -> Pipeline:
        """Create the scaler + ridge pipeline.

        :param config: Optional ``alpha`` (default 10.0).
        :return: An untrained pipeline.
        """
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", Ridge(alpha=config.get("alpha", 10.0), random_state=42)),
            ]
        )


@register_model("mlp_features")
class MlpFeatures(KerasModel):
    """Dense 29-32-16-1 network with ReLU activations on :class:`VectorFeatures`.

    Training uses Keras, but inference is a handful of NumPy matrix products: about two orders
    of magnitude faster per position than calling Keras, which matters inside a tree search.
    """

    extractor = VectorFeatures()
    architecture = "MLP 29 -> 32 -> 16 -> 1 (ReLU), NumPy inference"

    def __init__(self, network: Any):
        super().__init__(network)
        self._dense = None

    @classmethod
    def build_network(cls, config: Dict[str, Any]):
        """Create the dense network.

        :param config: Unused.
        :return: An uncompiled ``keras.Model``.
        """
        keras = _import_keras()
        features = keras.Input(shape=cls.extractor.input_shapes[0], name="features")
        hidden = keras.layers.Dense(32, activation="relu")(features)
        hidden = keras.layers.Dense(16, activation="relu")(hidden)
        return keras.Model(inputs=features, outputs=keras.layers.Dense(1)(hidden))

    def fit(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Train with Keras and refresh the cached NumPy weights.

        :return: Training summary (see :meth:`KerasModel.fit`).
        """
        summary = super().fit(*args, **kwargs)
        self._dense = None
        return summary

    def predict_target(self, inputs: Tuple[np.ndarray, ...]) -> np.ndarray:
        """Forward pass in NumPy.

        :param inputs: Encoded batch; one array of shape ``(n, 29)``.
        :return: Predictions, shape ``(n,)``.
        """
        if self._dense is None:
            self._dense = [
                (layer.get_weights()[0], layer.get_weights()[1])
                for layer in self.network.layers
                if layer.__class__.__name__ == "Dense"
            ]
        activations = np.asarray(inputs[0], dtype=np.float32)
        for index, (weights, bias) in enumerate(self._dense):
            activations = activations @ weights + bias
            if index < len(self._dense) - 1:
                activations = np.maximum(activations, 0.0)
        return activations.reshape(-1)


@register_model("cnn_resnet")
class CnnResnet(KerasModel):
    """Residual CNN over the piece planes, merged with a small dense branch for the extras.

    Trained with Huber loss (delta 0.1), Adam 3e-4, batch 256 and early stopping. The shipped
    weights come from the university course; retraining needs a GPU (about two minutes per
    epoch on ~2M positions).
    """

    extractor = BitboardFeatures()
    architecture = (
        "ResNet-style CNN (Conv32-64, two dilated residual blocks, Conv256, GAP) "
        "+ dense branch on 9 extras -> Dense256-128-64-1"
    )

    @classmethod
    def build_network(cls, config: Dict[str, Any]):
        """Create the residual network (572,353 parameters).

        :param config: Unused.
        :return: An uncompiled ``keras.Model`` with inputs ``[board, meta]``.
        """
        keras = _import_keras()
        layers = keras.layers

        def conv_bn_relu(tensor, filters, kernel, dilation=1):
            tensor = layers.Conv2D(
                filters, kernel, padding="same", dilation_rate=dilation, use_bias=False
            )(tensor)
            tensor = layers.BatchNormalization()(tensor)
            return layers.ReLU()(tensor)

        board_input = keras.Input(shape=(8, 8, 12), name="board")
        meta_input = keras.Input(shape=(9,), name="meta")

        x = conv_bn_relu(board_input, 32, 3)
        x = conv_bn_relu(x, 64, 3)

        residual = conv_bn_relu(x, 64, 3, dilation=2)
        x = layers.Add()([x, residual])

        shortcut = layers.BatchNormalization()(
            layers.Conv2D(128, 1, padding="same", use_bias=False)(x)
        )
        residual = conv_bn_relu(x, 128, 3, dilation=2)
        x = layers.Add()([shortcut, residual])

        x = conv_bn_relu(x, 256, 3)
        x = layers.GlobalAveragePooling2D()(x)

        meta = layers.Dense(128, use_bias=False)(meta_input)
        meta = layers.BatchNormalization()(meta)
        meta = layers.ReLU()(meta)
        meta = layers.Dense(64, activation="relu")(meta)

        z = layers.Concatenate()([x, meta])
        z = layers.Dense(256, use_bias=False)(z)
        z = layers.BatchNormalization()(z)
        z = layers.ReLU()(z)
        z = layers.Dense(128, activation="relu")(z)
        z = layers.Dense(64, activation="relu")(z)
        return keras.Model(inputs=[board_input, meta_input], outputs=layers.Dense(1)(z))


@register_model("cnn_light")
class CnnLight(KerasModel):
    """Lighter CNN (109,569 parameters) with the same inputs as ``cnn_resnet``.

    Only the saved weights survive from the course, so this model can be loaded and evaluated
    but not retrained.
    """

    extractor = BitboardFeatures()
    architecture = "CNN (5 conv layers 32-32-64-64-64, GAP) + dense branch -> Dense32-16-1"
    trainable = False
