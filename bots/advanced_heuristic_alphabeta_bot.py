import time
from typing import Callable

import chess

from adversarial_search.chess_game_state import ChessGameState
from adversarial_search.chess_problem import ChessProblem
from adversarial_search.game_algorithms import heuristic_alphabeta_iterative_deepening_search
from adversarial_search.game_problems import GameState
from bots.chess_bot import ChessBot
from bots.heuristic_alphabeta_bot import h3


class AdvancedHeuristicAlphaBetaBot(ChessBot):
    """
    Chess bot modifying heuristic alpha-beta search with configurable parameters.
    """

    def __init__(self, eval_function: Callable[[ChessGameState, chess.Color], float]=h3,
                 max_depth: int = 3, problem : ChessProblem = None) -> None:
        """
        Initialize the heuristic alpha-beta bot.

        :param eval_function: Evaluation function (state, player) -> float
        :type eval_function: Callable[[ChessGameState, chess.Color], float]
        :param max_depth: Maximum search depth
        :type max_depth: int
        """
        super().__init__("AdvancedHeuristicAlphaBetaBot", "Joaquin De Vicente Abad")
        self.eval_function = eval_function
        self.max_depth = max_depth
        self.problem = problem if problem is not None else ChessProblem(initial_board= None)

        self.max_turn_time = 0

        #Expandimos el diccionario antes de jugar para tener más consistencia y ahorrar tiempo de partida
        self.transposition_table = {i: None for i in range(100_000)}

        for i in range(100_000):
                del self.transposition_table[i]

    def get_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """
        Get the best move using heuristic alpha-beta search.

        :param board: Current chess position
        :type board: chess.Board
        :param time_limit: Maximum time to think in seconds
        :type time_limit: float
        :return: Best move found
        :rtype: chess.Move
        """
        state = ChessGameState(board= board)

        # Para modos de juego sin adición de tiempo
        if time_limit < 0.5:
            return state.get_legal_moves()[0]

        # Qué porcentaje del tiempo restante damos al movimiento
        move_fraction = 0.15
        self.max_turn_time = max(1.5, min(40.0, time_limit * move_fraction))

        move = heuristic_alphabeta_iterative_deepening_search(
                                            game= self.problem ,
                                            state= state,
                                            eval_fn= self.eval_function,
                                            cutoff_test= self.cutoff_test,
                                            transposition_table= self.transposition_table,
                                            max_depth= self.max_depth,
                                            time_limit= self.max_turn_time
                                            )

        return move


    def cutoff_test(self, state : GameState , depth_left : int, time_used : float) -> bool:

        #Intentamos asegurar la iteración una vez comenzada
        timeExtension = 1.5
        #Overhead estimation margin
        safety_margin = 0.3

        return depth_left == 0 or time_used >= (self.max_turn_time * timeExtension) - safety_margin