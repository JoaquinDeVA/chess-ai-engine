import keras
import chess
import numpy as np
import joblib


from adversarial_search.chess_problem import ChessProblem
from adversarial_search.game_algorithms import heuristic_alphabeta_search
from bots.chess_bot import ChessBot
from adversarial_search.chess_game_state import ChessGameState


class MLModel:

    # Valores asociados a las piezas (sin rey)
    piece_values: dict[int, float] = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }

    # Casillas centrales del tablero
    center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}

    MAX_MATERIAL = 78.0  # Material máximo sin contar reyes

    # Nuestro modelo
    model: object

    def __init__(self, model_file: str) -> None:
        self.model = joblib.load(model_file)

    def predict(self, state: ChessGameState) -> float:

        board: chess.Board = state.get_board()

        # Extract features
        features: np.ndarray = self.extract_features(board)

        # Linear / Ridge prediction
        value = float(self.model.predict(features.reshape(1, -1))[0])

        # Clip and scale to centipawns
        return np.clip(value, -1.0, 1.0) * 1267.0

    def extract_features(self, board: chess.Board) -> np.ndarray:
        """
        Convierte un tablero en un vector de características optimizado
        para modelos lineales / redes pequeñas.

        Características incluidas:
        - Turno
        - Fase de la partida
        - Material por color y diferencia
        - Conteo normalizado de piezas
        - Estructura de peones
        - Actividad del rey
        - Control del centro
        - Movilidad
        - Estado de jaque
        - Derechos de enroque
        """

        # Acumuladores
        material = [0.0, 0.0]                # [blanco, negro]
        piece_count = np.zeros((2, 5), dtype=np.float32)  # pawn..queen
        pawn_rank_sum = [0.0, 0.0]
        pawn_count = [0, 0]
        center_control = [0, 0]
        king_square = [None, None]

        # -------- ÚNICA pasada por el tablero --------
        for square, piece in board.piece_map().items():
            color = int(piece.color)
            ptype = piece.piece_type

            # Material y conteo (sin rey)
            if ptype != chess.KING:
                material[color] += self.piece_values[ptype]
                piece_count[color, ptype - 1] += 1

            # Peones
            if ptype == chess.PAWN:
                pawn_count[color] += 1
                pawn_rank_sum[color] += chess.square_rank(square) / 7.0

            # Centro
            if square in self.center_squares:
                center_control[color] += 1

            # Rey
            if ptype == chess.KING:
                king_square[color] = square

        # -------- Features derivadas --------

        # Diferencia de material
        material_diff = material[0] - material[1]

        # Fase de la partida (0 = final, 1 = apertura)
        phase = (material[0] + material[1]) / self.MAX_MATERIAL

        # Avance medio de peones
        pawn_rank_avg = [
            pawn_rank_sum[c] / pawn_count[c] if pawn_count[c] > 0 else 0.0
            for c in (0, 1)
        ]

        # Actividad del rey (distancia al centro)
        def king_activity(square):
            if square is None:
                return 0.0
            file_dist = abs(chess.square_file(square) - 3.5)
            rank_dist = abs(chess.square_rank(square) - 3.5)
            return (file_dist + rank_dist) / 7.0

        king_activity_white = king_activity(king_square[0])
        king_activity_black = king_activity(king_square[1])

        # Movilidad normalizada
        mobility = board.legal_moves.count() / 50.0

        # -------- Vector final (normalizado) --------
        vector = np.array([
            1.0 if board.turn == chess.WHITE else -1.0,  # Turno

            phase,                                      # Fase
            1.0 - phase,

            material[0] / 39.0,                         # Material blanco
            material[1] / 39.0,                         # Material negro
            material_diff / 39.0,                       # Diferencia material

            *(piece_count[0] / np.array([8, 2, 2, 2, 1])),  # Piezas blancas
            *(piece_count[1] / np.array([8, 2, 2, 2, 1])),  # Piezas negras

            pawn_rank_avg[0],                           # Avance peones blanco
            pawn_rank_avg[1],                           # Avance peones negro
            pawn_count[0] / 8.0,                        # Nº peones blanco
            pawn_count[1] / 8.0,                        # Nº peones negro

            center_control[0] / 4.0,                    # Centro blanco
            center_control[1] / 4.0,                    # Centro negro

            king_activity_white,                        # Actividad rey blanco
            king_activity_black,                        # Actividad rey negro

            board.is_check() and board.turn == chess.WHITE,
            board.is_check() and board.turn == chess.BLACK,

            board.has_kingside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),

            mobility                                    # Movilidad
        ], dtype=np.float32)

        return vector

















class MLBot(ChessBot):
    def __init__(
        self,
        model_file: str = "bots/ml/regresor.pkl",
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
        """Get the best move using heuristic alpha-beta search.

        :param board: Current chess position
        :type board: chess.Board
        :param time_limit: Maximum time to think in seconds
        :type time_limit: float
        :return: Best move found
        :rtype: chess.Move
        """
        chess_state = ChessGameState(board)
        chess_problem = ChessProblem(board)


        try:
            # Usamos heuristic alpha-beta search
            move, _ = heuristic_alphabeta_search(
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

