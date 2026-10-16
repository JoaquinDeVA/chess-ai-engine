"""``linear_baseline``: the smallest possible model, used to demonstrate the extension flow.

Ordinary least squares on six material numbers (see :class:`models.features.MaterialFeatures`).
Training takes seconds and gives a lower bound for what the larger models must beat.
"""

from typing import Any, Dict

from sklearn.linear_model import LinearRegression

from .features import MaterialFeatures
from .registry import register_model
from .sklearn_model import SklearnModel


@register_model("linear_baseline")
class LinearBaseline(SklearnModel):
    """Linear regression on side to move and piece-count differences."""

    extractor = MaterialFeatures()
    architecture = "LinearRegression on 6 material features (turn + piece-count differences)"

    @classmethod
    def build_estimator(cls, config: Dict[str, Any]) -> LinearRegression:
        """Create the regressor.

        :param config: Unused; the model has no hyper-parameters.
        :return: An untrained ``LinearRegression``.
        """
        return LinearRegression()
