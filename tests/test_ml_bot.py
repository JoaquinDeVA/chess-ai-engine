import unittest
import chess
import numpy as np
import os
from bots.ml_bot import MLModel, MLBot
from adversarial_search.chess_game_state import ChessGameState

class TestMLChessBot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up paths and model once for all tests"""

        # Resolve the absolute path to the model file relative to this test script
        cls.model_file = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),  # folder of this test script
                "..", "bots", "ml", "chess_ML.keras"
            )
        )

        # Debug: ensure the file exists
        print("Model file path:", cls.model_file)
        if not os.path.exists(cls.model_file):
            raise FileNotFoundError(f"Model file not found: {cls.model_file}")

        # Initialize model and bot
        cls.ml_model = MLModel(cls.model_file)
        cls.bot = MLBot(model_file=cls.model_file, max_depth=1)

    def setUp(self):
        """Set up a fresh board for each test"""
        self.board = chess.Board()
        self.state = ChessGameState(self.board)

    def test_split_dims(self):
        """Test that split_dims returns a tensor of correct shape and type"""
        tensor = self.ml_model.split_dims(self.board)
        self.assertEqual(tensor.shape, (8, 8, 12))
        self.assertTrue(np.issubdtype(tensor.dtype, np.integer))
        self.assertTrue(np.any(tensor != 0))  # ensure board has pieces

    def test_extract_features(self):
        """Test feature extraction returns correct size and values"""
        features = self.ml_model.extract_features(self.board)
        self.assertEqual(features.shape, (9,))
        self.assertTrue(np.issubdtype(features.dtype, np.floating))
        self.assertEqual(features[0], 1.0)  # turn feature for white

    def test_predict(self):
        """Test that the model returns a float prediction"""
        prediction = self.ml_model.predict(self.state)
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, -20000)
        self.assertLessEqual(prediction, 20000)

    def test_evaluate_position(self):
        """Test bot evaluation returns float between 0 and 1"""
        eval_white = self.bot.evaluate_position(self.state, chess.WHITE)
        eval_black = self.bot.evaluate_position(self.state, chess.BLACK)
        self.assertTrue(0 <= eval_white <= 1)
        self.assertTrue(0 <= eval_black <= 1)

    def test_get_move(self):
        """Test bot returns a legal move"""
        move = self.bot.get_move(self.board, time_limit=1.0)
        self.assertIsInstance(move, chess.Move)
        self.assertIn(move, self.board.legal_moves)

if __name__ == "__main__":
    unittest.main()
