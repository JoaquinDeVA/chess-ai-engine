"""Tests of the move-ranking test set builder, with a fake engine."""

import chess
import chess.engine

from chess_lab.datasets.build_move_ranking import eligible, label_position

MATE_IN_ONE = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"


class FakeEngine:
    """Answers every analysis with one line per legal move, the first one a mate."""

    def analyse(self, board, limit, multipv):
        moves = list(board.legal_moves)
        assert multipv == len(moves)
        infos = []
        for i, move in enumerate(moves):
            score = chess.engine.Mate(1) if i == 0 else chess.engine.Cp(10 * i)
            infos.append({"pv": [move], "score": chess.engine.PovScore(score, board.turn)})
        return infos


def test_label_position_gives_one_row_per_legal_move():
    rows = label_position(FakeEngine(), MATE_IN_ONE, depth=1)
    assert len(rows) == chess.Board(MATE_IN_ONE).legal_moves.count()
    move, cp, mate = rows[0]
    assert (cp != cp) and mate == 1.0  # NaN centipawns on a mate line
    assert rows[1][2] != rows[1][2] and rows[1][1] == 10.0


def test_eligible_skips_finished_and_single_move_positions():
    fool = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4#"):
        fool.push_san(san)
    one_move = "7k/8/8/8/8/8/6q1/7K w - - 0 1"  # only Kxg2
    assert chess.Board(one_move).legal_moves.count() == 1
    fens = [fool.fen(), one_move, chess.STARTING_FEN]
    assert eligible(fens, 5) == [chess.STARTING_FEN]
    assert eligible(fens * 3, 2) == [chess.STARTING_FEN] * 2
