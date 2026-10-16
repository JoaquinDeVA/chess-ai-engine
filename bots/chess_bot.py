"""
Chess Bot Base Class

Defines the abstract interface that every chess bot implements; subclass :class:`ChessBot`
(or one of the search bases in ``bots/_search_bots.py``) to add a new engine.

Based on the AI course framework (Inteligencia Artificial e Ingeniería del Conocimiento,
CEU San Pablo University).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import chess


class ChessBot(ABC):
    """
    Abstract base class for all chess bots.

    All chess bots must inherit from this class and implement the get_move method.
    """
    
    def __init__(self, name: str, author: str = "Unknown"):
        """
        Initialize the chess bot.

        :param name: Name of your bot (should be unique)
        :type name: str
        :param author: Name of the bot author
        :type author: str
        """
        self.name = name
        self.author = author

    @abstractmethod
    def get_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """
        Get the best move for the current position.

        :param board: Current chess position (python-chess Board object)
        :type board: chess.Board
        :param time_limit: Maximum time to think in seconds
        :type time_limit: float
        :return: Your chosen move (must be legal)
        :rtype: chess.Move
        :raises: Should not raise exceptions - return any legal move if in trouble
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this bot.

        :return: Dict with bot information
        :rtype: Dict[str, Any]
        """
        return {
            "name": self.name,
            "author": self.author,
            "description": "A chess bot",
            "version": "1.0",
        }
    
    def on_game_start(self, color: chess.Color, opponent_name: str) -> None:
        """
        Called when a new game starts.

        Override this method if your bot needs to:
        - Reset internal state between games
        - Prepare strategy based on opponent or color
        - Initialize game-specific data structures

        :param color: chess.WHITE or chess.BLACK - your color this game
        :type color: chess.Color
        :param opponent_name: Name of your opponent
        :type opponent_name: str
        """
        pass

    def on_game_end(self, result: str, board: chess.Board) -> None:
        """
        Called when a game ends.

        Override this method if your bot needs to:
        - Learn from the game result (for RL bots)
        - Update internal statistics
        - Save game data for analysis

        :param result: Game result ("1-0", "0-1", "1/2-1/2")
        :type result: str
        :param board: Final board position
        :type board: chess.Board
        """
        pass
    
    def __str__(self) -> str:
        """
        String representation of the bot.

        :return: String representation
        :rtype: str
        """
        return f"{self.name} by {self.author}"
    
    def __repr__(self) -> str:
        """
        Developer representation of the bot.

        :return: Developer representation
        :rtype: str
        """
        return f"ChessBot(name='{self.name}', author='{self.author}')"
