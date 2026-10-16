from typing import Dict, Any

import chess

from bots.chess_bot import ChessBot


class HumanPlayer(ChessBot):
    """
    Human player interface for single-player games.

    This handles getting moves from human players through the console.
    """

    def __init__(self, name: str = "Human Player"):
        """
        Initialize human player.

        :param name: Name for the human player
        :type name: str
        """
        super().__init__(name, "Chess AI Engine")

    def get_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """
        Get move from human player through console input.

        :param board: Current chess position
        :type board: chess.Board
        :param time_limit: Time limit (ignored for human players)
        :type time_limit: float
        :return: Move chosen by human player
        :rtype: chess.Move
        """
        print(f"Legal moves: {[move.uci() for move in board.legal_moves]}")

        while True:
            try:
                move_str = input("Enter your move (e.g., e2e4): ").strip()
                move = chess.Move.from_uci(move_str)

                if move in board.legal_moves:
                    return move
                else:
                    print("Illegal move! Try again.")

            except (ValueError, KeyboardInterrupt):
                print("Invalid move format! Use format like 'e2e4'")

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about human player.

        :return: Human player information dictionary
        :rtype: Dict[str, Any]
        """
        info = super().get_info()
        info.update({
            "description": "Human player",
            "version": "1.0",
        })
        return info
