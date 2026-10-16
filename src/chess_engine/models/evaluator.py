"""Adapter that lets a search use an :class:`EvaluationModel` as its evaluation function."""

import chess

from chess_engine.search.chess_game_state import ChessGameState

from .base import EXPECTED_SCORE_MARGIN, EvaluationModel


class ModelEvaluator:
    """Callable ``(state, player) -> expected score in [0.01, 0.99]`` backed by a model.

    :param model: Model that evaluates positions from White's point of view.
    """

    def __init__(self, model: EvaluationModel):
        self.model = model

    def __call__(self, state: ChessGameState, player: chess.Color) -> float:
        """Expected score of ``player`` in ``state``.

        :param state: Position to evaluate.
        :param player: Player whose expected score is returned.
        :return: Value in ``[0.01, 0.99]``: 0.5 is equal, higher is better for ``player``.
        """
        score = self.model.expected_score(state.get_board())
        if player != chess.WHITE:
            score = 1.0 - score
        return min(1.0 - EXPECTED_SCORE_MARGIN, max(EXPECTED_SCORE_MARGIN, score))
