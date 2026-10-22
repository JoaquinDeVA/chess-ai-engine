"""Tests of the endgame tablebase wrapper, with the five 3-piece Syzygy tables in tests/data/syzygy."""

import random
from pathlib import Path

import chess
import pytest

from chess_engine.bots.iterative_heuristic_bot import IterativeHeuristicBot
from chess_engine.knowledge.endgame_tablebase import EndgameTablebase

TABLES = Path(__file__).resolve().parent / "data" / "syzygy"


@pytest.fixture(scope="module")
def tablebase():
    tb = EndgameTablebase(TABLES)
    yield tb
    tb.close()


def random_position(rng, white_extra, black_extra=""):
    """A legal position with both kings, White's extra pieces and Black's, White to move."""
    while True:
        board = chess.Board(None)
        squares = rng.sample(range(64), 2 + len(white_extra) + len(black_extra))
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        pieces = [(c, chess.WHITE) for c in white_extra] + [(c, chess.BLACK) for c in black_extra]
        for square, (symbol, color) in zip(squares[2:], pieces, strict=True):
            board.set_piece_at(
                square, chess.Piece(chess.Piece.from_symbol(symbol).piece_type, color)
            )
        board.turn = chess.WHITE
        if board.is_valid() and not board.is_game_over():
            return board


def test_missing_directory_raises_and_open_if_present_returns_none(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        EndgameTablebase(missing)
    assert EndgameTablebase.open_if_present(missing) is None


def test_covers_only_positions_with_at_most_five_pieces_and_no_castling(tablebase):
    assert tablebase.covers(chess.Board("8/8/8/8/8/2k5/8/K6Q w - - 0 1"))
    assert not tablebase.covers(chess.Board())
    assert not tablebase.covers(chess.Board("r3k3/8/8/8/8/8/8/R3K3 w KQ - 0 1"))


def test_no_move_for_uncovered_or_finished_positions(tablebase):
    assert tablebase.best_move(chess.Board()) is None
    assert (
        tablebase.best_move(chess.Board("7k/5Q1K/8/8/8/8/8/8 b - - 0 1")) is None
    )  # stalemate? no: mate
    assert tablebase.best_move(chess.Board("8/8/8/8/8/2k5/8/K7 w - - 0 1")) is None  # bare kings


def test_a_mate_in_one_is_played(tablebase):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    move = tablebase.best_move(board)
    board.push(move)
    assert board.is_checkmate()


@pytest.mark.parametrize("extra", ["Q", "R"])
def test_the_side_with_the_extra_piece_mates_in_a_few_moves_against_the_defender(tablebase, extra):
    rng = random.Random(7)
    for _ in range(6):
        board = random_position(rng, extra)
        for _ in range(40):
            if board.is_game_over():
                break
            board.push(tablebase.best_move(board))
        assert board.is_checkmate(), board.fen()


def test_the_best_move_never_changes_the_theoretical_result(tablebase):
    rng = random.Random(3)
    checked = 0
    for white, black in [("P", ""), ("", "P"), ("R", ""), ("", "R"), ("Q", ""), ("", "Q")]:
        for _ in range(12):
            board = random_position(rng, white, black)
            if not tablebase.covers(board):
                continue
            before = tablebase._tables.probe_wdl(board)
            move = tablebase.best_move(board)
            board.push(move)
            if board.is_game_over():
                continue
            after = -tablebase._tables.probe_wdl(board)
            assert after == before, (board.fen(), before, after)
            checked += 1
    assert checked > 20


def test_the_bot_plays_the_tablebase_move_without_searching(tablebase):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    bot = IterativeHeuristicBot(tablebase=tablebase)
    move = bot.get_move(board, 60.0)
    assert move == tablebase.best_move(board)
    assert bot.last_search_stats.stop_reason == "tablebase"
    assert bot.last_search_stats.nodes_visited == 0


def test_the_bot_without_tablebase_searches(tablebase):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    bot = IterativeHeuristicBot(tablebase=None)
    bot.get_move(board, 5.0)
    assert bot.last_search_stats.stop_reason != "tablebase"
    assert bot.last_search_stats.nodes_visited > 0
