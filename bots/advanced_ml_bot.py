import chess
import json
import chess.syzygy
import keras
import numpy as np
import joblib
import time
import random
from abc import ABC, abstractmethod
from typing import Any, Optional, Hashable, Dict, List, Callable, NamedTuple

from adversarial_search.chess_game_state import ChessGameState
from adversarial_search.chess_problem import ChessProblem
from adversarial_search.game_problems import GameProblem, GameState
from bots.chess_bot import ChessBot


class AdversarialSearchResult(NamedTuple):
    value: float
    move: Any | None


class TTEntry:
    def __init__(self, result : AdversarialSearchResult, depth_remaining: int, flag: str):
        self.result = result
        self.depth_remaining = depth_remaining
        self.flag = flag


def reorder_by_killer_moves(moves: list[Any], depth: int, diccionario : Dict[int, List[Any]]):
    killer_moves = diccionario.get(depth, [])
    moves.sort(key= lambda move: move not in killer_moves)
    return moves


def advanced_heuristic_alphabeta_search(
    game: GameProblem,
    state: GameState,
    eval_fn: Callable[[GameState, Any], float],
    cutoff_test: Callable[[GameState, int, float], bool],
    killer_moves: Dict[int, List[chess.Move]] = None,
    transposition_table: Dict[Hashable, TTEntry] = None,
    order_moves_callback: Callable[[chess.Board, list[chess.Move]], list[chess.Move]] | None = None,
    max_depth: int = 3
) -> Any:
    root_player = state.get_current_player()
    time_of_start = time.time()

    epsilon = 0.003

    if killer_moves is None:
        killer_moves = {}

    if transposition_table is None:
        transposition_table = {}

    def max_value(
        state: "GameState", alpha: float, beta: float, depth_remaining: int
    ) -> AdversarialSearchResult:
        key = state.key()
        alpha_orig = alpha
        beta_orig = beta

        entry = transposition_table.get(key)

        if entry and entry.depth_remaining >= depth_remaining:
            if entry.flag == 'EXACT':
                return entry.result
            elif entry.flag == 'LOWERBOUND':
                alpha = max(alpha, entry.result.value)
            elif entry.flag == 'UPPERBOUND':
                beta = min(beta, entry.result.value)
            if alpha >= beta:
                return entry.result

        if game.is_terminal(state):
            utility = game.utility(state, root_player)

            if utility == 1:
                utility -= (max_depth - depth_remaining) * epsilon
            elif utility == 0:
                utility += (max_depth - depth_remaining) * epsilon

            return AdversarialSearchResult(utility, None)

        if cutoff_test(state, depth_remaining, time.time() - time_of_start):
            return AdversarialSearchResult(eval_fn(state, root_player), None)

        best_value = float("-inf")
        best_move = None

        killer_moves.setdefault(depth_remaining, [])
        actions = reorder_by_killer_moves(game.actions(state), depth_remaining, killer_moves)

        if order_moves_callback:
            actions = order_moves_callback(state.get_board() , actions)

        for action in actions:
            result = min_value(game.result(state, action), alpha, beta, depth_remaining - 1)

            if result.value > best_value:
                best_value = result.value
                best_move = action
                alpha = max(alpha, best_value)

            if alpha >= beta:
                if action not in killer_moves[depth_remaining]:
                    killer_moves[depth_remaining].append(action)
                if len(killer_moves[depth_remaining]) > 3:
                    killer_moves[depth_remaining].pop(0)
                break

        if best_value <= alpha_orig:
            flag = 'UPPERBOUND'
        elif best_value >= beta_orig:
            flag = 'LOWERBOUND'
        else:
            flag = 'EXACT'

        transposition_table[key] = TTEntry(
            result=AdversarialSearchResult(best_value, best_move),
            depth_remaining=depth_remaining,
            flag=flag
        )

        return AdversarialSearchResult(best_value, best_move)

    def min_value(
        state: "GameState", alpha: float, beta: float, depth_remaining: int
    ) -> AdversarialSearchResult:
        key = state.key()
        alpha_orig = alpha
        beta_orig = beta

        entry = transposition_table.get(key)

        if entry and entry.depth_remaining >= depth_remaining:
            if entry.flag == 'EXACT':
                return entry.result
            elif entry.flag == 'LOWERBOUND':
                alpha = max(alpha, entry.result.value)
            elif entry.flag == 'UPPERBOUND':
                beta = min(beta, entry.result.value)
            if alpha >= beta:
                return entry.result

        if game.is_terminal(state):
            utility = game.utility(state, root_player)

            if utility == 1:
                utility -= depth_remaining * epsilon
            elif utility == 0:
                utility += depth_remaining * epsilon

            return AdversarialSearchResult(utility, None)

        if cutoff_test(state, depth_remaining, time.time() - time_of_start):
            return AdversarialSearchResult(eval_fn(state, root_player), None)

        best_value = float("inf")
        best_move = None

        killer_moves.setdefault(depth_remaining, [])
        actions = reorder_by_killer_moves(game.actions(state), depth_remaining, killer_moves)

        if order_moves_callback:
            actions = order_moves_callback(state.get_board() , actions)

        for action in actions:
            result = max_value(game.result(state, action), alpha, beta, depth_remaining - 1)

            if result.value < best_value:
                best_value = result.value
                best_move = action
                beta = min(beta, best_value)

            if alpha >= beta:
                if action not in killer_moves[depth_remaining]:
                    killer_moves[depth_remaining].append(action)
                if len(killer_moves[depth_remaining]) > 3:
                    killer_moves[depth_remaining].pop(0)
                break

        if best_value <= alpha_orig:
            flag = 'UPPERBOUND'
        elif best_value >= beta_orig:
            flag = 'LOWERBOUND'
        else:
            flag = 'EXACT'

        transposition_table[key] = TTEntry(
            result=AdversarialSearchResult(best_value, best_move),
            depth_remaining=depth_remaining,
            flag=flag
        )

        return AdversarialSearchResult(best_value, best_move)

    search_result = max_value(state, float("-inf"), float("inf"), depth_remaining= max_depth)
    return search_result.move











def heuristic_alphabeta_iterative_deepening_search(
    game: GameProblem,
    state: GameState,
    eval_fn: Callable[[GameState, Any], float],
    cutoff_test: Callable[[GameState, int, float], bool],
    killer_moves: Dict[int, List[chess.Move]] = None,
    transposition_table: Dict[Hashable, TTEntry] = None,
    order_moves_callback: Callable[[chess.Board, list[chess.Move]], list[chess.Move]] | None = None,
    max_depth: int = 3,
    time_limit: float = 2,
) -> Any:
    start_time = time.time()
    overshoot_factor = 1.15

    if killer_moves is None:
        killer_moves = {}

    if transposition_table is None:
        transposition_table = {}

    depth_times = []

    for depth in range(1, max_depth + 1):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed

        if len(depth_times) > 2:
            last = depth_times[-1]
            prev = depth_times[-2]

            if prev > 1e-6 and last > 1e-6:
                branching_factor_est = last / prev
            else:
                branching_factor_est = 1.5

            time_estimation = last * branching_factor_est

            if time_estimation  > remaining * overshoot_factor:
                break

        depth_timer = time.time()

        result = advanced_heuristic_alphabeta_search(
                                          game= game,
                                          state= state,
                                          eval_fn= eval_fn,
                                          cutoff_test= cutoff_test,
                                          killer_moves= killer_moves,
                                          transposition_table= transposition_table,
                                          order_moves_callback= order_moves_callback,
                                          max_depth= depth,
                                          )

        depth_times.append(time.time() - depth_timer)

        if (time.time() - start_time) >= time_limit:
            break

    return result











class MLModel:

    piece_values: dict[int, float] = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }

    center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
    MAX_MATERIAL = 78.0

    model: object

    def __init__(self, model_file: str) -> None:
        self.model = joblib.load(model_file)

    def predict(self, state: ChessGameState) -> float:
        board: chess.Board = state.get_board()
        features: np.ndarray = self.extract_features(board)
        value = float(self.model.predict(features.reshape(1, -1))[0])
        return np.clip(value, -1.0, 1.0) * 1267.0

    def extract_features(self, board: chess.Board) -> np.ndarray:
        material = [0.0, 0.0]
        piece_count = np.zeros((2, 5), dtype=np.float32)
        pawn_rank_sum = [0.0, 0.0]
        pawn_count = [0, 0]
        center_control = [0, 0]
        king_square = [None, None]

        for square, piece in board.piece_map().items():
            color = int(piece.color)
            ptype = piece.piece_type

            if ptype != chess.KING:
                material[color] += self.piece_values[ptype]
                piece_count[color, ptype - 1] += 1

            if ptype == chess.PAWN:
                pawn_count[color] += 1
                pawn_rank_sum[color] += chess.square_rank(square) / 7.0

            if square in self.center_squares:
                center_control[color] += 1

            if ptype == chess.KING:
                king_square[color] = square

        material_diff = material[0] - material[1]
        phase = (material[0] + material[1]) / self.MAX_MATERIAL

        pawn_rank_avg = [
            pawn_rank_sum[c] / pawn_count[c] if pawn_count[c] > 0 else 0.0
            for c in (0, 1)
        ]

        def king_activity(square):
            if square is None:
                return 0.0
            file_dist = abs(chess.square_file(square) - 3.5)
            rank_dist = abs(chess.square_rank(square) - 3.5)
            return (file_dist + rank_dist) / 7.0

        king_activity_white = king_activity(king_square[0])
        king_activity_black = king_activity(king_square[1])

        mobility = board.legal_moves.count() / 50.0

        vector = np.array([
            1.0 if board.turn == chess.WHITE else -1.0,
            phase,
            1.0 - phase,
            material[0] / 39.0,
            material[1] / 39.0,
            material_diff / 39.0,
            *(piece_count[0] / np.array([8, 2, 2, 2, 1])),
            *(piece_count[1] / np.array([8, 2, 2, 2, 1])),
            pawn_rank_avg[0],
            pawn_rank_avg[1],
            pawn_count[0] / 8.0,
            pawn_count[1] / 8.0,
            center_control[0] / 4.0,
            center_control[1] / 4.0,
            king_activity_white,
            king_activity_black,
            board.is_check() and board.turn == chess.WHITE,
            board.is_check() and board.turn == chess.BLACK,
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            mobility
        ], dtype=np.float32)

        return vector


class MLBot(ChessBot):
    def __init__(
        self,
        model_file: str,
        max_depth: int = 3,
        name = "MLBot"
    ):
        super().__init__(name= name, author = "Joaquin De Vicente Abad")
        self.ml_model = MLModel(model_file)
        self.max_depth = max_depth

    def evaluate_position(self, state: ChessGameState, player: chess.Color) -> float:
        prediccion = self.ml_model.predict(state)
        prediccion = prediccion if player == chess.WHITE else -prediccion
        return float ((prediccion / 20000.0 + 1.0 )/ 2.0)

    def get_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        chess_state = ChessGameState(board)
        chess_problem = ChessProblem(board)

        try:
            move, _ = heuristic_alphabeta_iterative_deepening_search(
                chess_problem,
                chess_state,
                self.evaluate_position,
                self.cutoff_test,
            )
            return move
        except Exception as e:
            raise e

    def cutoff_test(self,state: ChessGameState, depth: int, elapsed_time: float) -> bool:
        return depth >= self.max_depth


class AdvancedMLBot(MLBot):

    def __init__(
        self,
        model_file: str = "bots/advanced_ml_files/ml/regresor.pkl",
        oppening_file: str = "bots/advanced_ml_files/BBDD/oppening/best_moves.json",
        max_depth: int = 3,
        name = "AdvancedMLBot",
    ):
        super().__init__(model_file=model_file, max_depth=max_depth, name=name)

        self.max_turn_time = None
        self.transposition_table = {}
        self.problem = ChessProblem(initial_board= None)


        with open(oppening_file, "r", encoding="utf-8") as f:
            self.oppening : dict = json.load(f)

    def get_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        state = ChessGameState(board=board)

        if self.max_turn_time is None:
            if state.get_fen() in self.oppening:
                move_notation = self.oppening[state.get_fen()]
                return chess.Move.from_uci(move_notation)

        move_fraction = 0.15
        self.max_turn_time = max(1.5, min(40.0, time_limit * move_fraction))

        move = heuristic_alphabeta_iterative_deepening_search(
            game = self.problem,
            state=state,
            eval_fn=self.evaluate_position,
            cutoff_test=self.cutoff_test,
            transposition_table=self.transposition_table,
            max_depth=self.max_depth,
            time_limit=self.max_turn_time
        )

        return move


    def cutoff_test(self, state : GameState , depth_left : int, time_used : float) -> bool:
        time_extension = 1.5
        safety_margin = 0.3

        return depth_left == 0 or time_used >= (self.max_turn_time * time_extension) - safety_margin

    def order_moves_callback(self, board : chess.Board, moves : list[chess.Move]) -> list[chess.Move]:
        """
        Reordenación simple por MVV
        """
        def mvv_lva_score(move: chess.Move) -> int:

            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)

            if victim is None or attacker is None:
                return 0

            # MVV-LVA: high victim value, low attacker value
            return 10 * victim.piece_type - attacker.piece_type

        captures = []
        quiets = []

        for move in moves:
            if board.is_capture(move):
                captures.append(move)
            else:
                quiets.append(move)

        # Reordenamos los movimientos que son capturas, mantenemos orden previo para no olvidar killer moves.
        captures.sort(key=mvv_lva_score, reverse=True)

        return captures + quiets
