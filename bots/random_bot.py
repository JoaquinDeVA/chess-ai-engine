import random
from typing import Dict, Any

import chess

from bots.chess_bot import ChessBot


class RandomBot(ChessBot):
    """
    Simple random move bot that chooses moves randomly from all legal options.
    """

    def __init__(self):
        """
        Initialize the random bot.
        """
        super().__init__("RandomBot", "AI Course")
        self.random = random.Random(30)


    def get_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """
        Return a random legal move.

        :param board: Current chess position
        :type board: chess.Board
        :param time_limit: Maximum time to think in seconds (ignored)
        :type time_limit: float
        :return: A randomly chosen legal move
        :rtype: chess.Move
        """
        # Get all legal moves
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            # This shouldn't happen if the game isn't over, but be safe
            return chess.Move.null()

        # Pick a random move
        chosen_move = self.random.choice(legal_moves)

        return chosen_move

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about RandomBot.

        :return: Bot information dictionary
        :rtype: Dict[str, Any]
        """
        info = super().get_info()
        info.update({
            "description": "Chooses random legal moves",
            "version": "1.0",
        })
        return info