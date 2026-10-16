"""Machine-learning bot: fixed-depth alpha-beta whose evaluation comes from a trained model."""

from typing import Union

import chess

from adversarial_search.chess_game_state import ChessGameState
from bots._search_bots import AUTHOR, FixedDepthBot
from models import EvaluationModel, load_model
from models.evaluator import ModelEvaluator

#: Model used when none is given. Train it with ``python -m training.train --model mlp_features``.
DEFAULT_MODEL = "mlp_features"


class MLBot(FixedDepthBot):
    """Alpha-beta to a fixed depth, evaluating leaves with a learned model.

    :param model: Registered model name, or an already loaded :class:`EvaluationModel`.
    :param max_depth: Plies to search.
    :param name: Bot name shown in games and tournaments.
    :raises FileNotFoundError: If the saved model does not exist yet.
    """

    def __init__(
        self,
        model: Union[str, EvaluationModel] = DEFAULT_MODEL,
        max_depth: int = 3,
        name: str = "MLBot",
    ):
        super().__init__(name, AUTHOR, max_depth)
        self.ml_model = load_model(model) if isinstance(model, str) else model
        self.evaluator = ModelEvaluator(self.ml_model)

    def evaluate(self, state: ChessGameState, player: chess.Color) -> float:
        """Expected score of ``player`` according to the model.

        :param state: Position to evaluate.
        :param player: Player the value is for.
        :return: Value in ``[0.01, 0.99]``.
        """
        return self.evaluator(state, player)

    def get_info(self):
        """Describe the bot.

        :return: Bot information including the model in use.
        """
        info = super().get_info()
        info.update(
            description=f"Fixed-depth alpha-beta evaluated by model '{self.ml_model.name}'",
            model=self.ml_model.name,
            max_depth=self.max_depth,
        )
        return info
