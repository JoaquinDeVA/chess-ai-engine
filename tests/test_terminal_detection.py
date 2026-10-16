"""``ChessGameState`` must agree with ``board.outcome(claim_draw=True)`` at every position."""

import random

import chess

from chess_engine.search.chess_game_state import ChessGameState


def reference_winner(board: chess.Board):
    """Winner as ``ChessGameState.get_winner`` reports it, from python-chess."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return None
    return "white" if outcome.winner == chess.WHITE else "black"


def assert_equivalent(state: ChessGameState) -> None:
    """Compare the state with python-chess on the board it wraps."""
    board = state.get_board()
    expected = board.outcome(claim_draw=True)
    assert state.is_over() == (expected is not None), board.fen()
    assert state.get_winner() == reference_winner(board), board.fen()
    assert state._outcome == expected, board.fen()


def termination(state: ChessGameState):
    """Termination of the state's outcome (None if the game is not over)."""
    state.is_over()
    return state._outcome.termination if state._outcome else None


def play_game(fen: str, seed: int, plies: int, shuffle: float) -> int:
    """Play a random game checking every ply, three ways of building the state.

    :param fen: Start position.
    :param seed: Random seed.
    :param plies: Maximum number of plies.
    :param shuffle: Probability of preferring a king or knight move (provokes repetitions).
    :return: Number of plies played.
    """
    rng = random.Random(seed)
    board = chess.Board(fen)
    chained = ChessGameState(board)
    for ply in range(plies):
        assert_equivalent(chained)
        assert_equivalent(ChessGameState(board))
        if board.outcome(claim_draw=False) is not None:
            return ply
        moves = list(board.legal_moves)
        quiet = [
            m
            for m in moves
            if board.piece_type_at(m.from_square) in (chess.KING, chess.KNIGHT)
            and not board.is_capture(m)
        ]
        move = rng.choice(quiet) if quiet and rng.random() < shuffle else rng.choice(moves)
        board.push(move)
        # Like the search: a copy of the parent receives the move.
        chained = chained.copy().make_move(move)
    assert_equivalent(chained)
    return plies


def test_sparse_positions_with_many_repetitions():
    fens = [
        "8/8/4k3/8/8/4K3/8/R7 w - - 0 1",
        "4k3/8/8/8/8/8/8/1N2K1N1 w - - 0 1",
        "r3k3/8/8/8/8/8/8/R3K1N1 w Qq - 0 1",
        "4k3/pp6/8/8/8/8/PP6/4K1N1 w - - 0 1",
    ]
    for seed in range(40):
        play_game(fens[seed % len(fens)], seed, plies=260, shuffle=0.9)


def test_normal_random_games():
    for seed in range(15):
        play_game(chess.STARTING_FEN, 1000 + seed, plies=200, shuffle=0.3)


def test_states_with_history_built_directly_from_a_board():
    rng = random.Random(7)
    for _ in range(20):
        board = chess.Board("4k3/8/8/8/8/8/8/R3K1N1 w - - 0 1")
        for _ in range(rng.randrange(0, 60)):
            moves = [m for m in board.legal_moves if not board.is_capture(m)]
            if board.outcome(claim_draw=False) is not None or not moves:
                break
            board.push(rng.choice(moves))
        assert_equivalent(ChessGameState(board))


def test_threefold_by_occurrence():
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        board.push_uci(uci)
    state = ChessGameState(board)
    assert state.is_over()
    assert state.get_winner() is None
    assert termination(state) == chess.Termination.THREEFOLD_REPETITION


def test_threefold_claimable_by_the_next_move():
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
        board.push_uci(uci)
    assert not board.is_repetition(3)
    state = ChessGameState(board)
    assert state.is_over()
    assert termination(state) == chess.Termination.THREEFOLD_REPETITION
    # One move earlier no legal move repeats a position for the third time.
    board.pop()
    assert not ChessGameState(board).is_over()


def test_repetition_interrupted_by_a_capture():
    board = chess.Board("4k3/8/8/3p4/4P3/8/8/R3K1N1 w - - 0 1")
    for uci in ["g1f3", "e8e7", "f3g1", "e7e8", "g1f3", "e8e7", "f3g1"]:
        board.push_uci(uci)
    assert ChessGameState(board).is_over()
    for uci in ["e7e8", "e4d5"]:  # capture: earlier positions no longer count
        board.push_uci(uci)
    for uci in ["e8e7", "g1f3", "e7e8", "f3g1"]:
        board.push_uci(uci)
    state = ChessGameState(board)
    assert not state.is_over()
    assert_equivalent(state)


def test_repetition_interrupted_by_a_pawn_move_through_make_move():
    state = ChessGameState(chess.Board("4k3/8/8/8/8/8/P7/R3K1N1 w - - 0 1"))
    for uci in ["g1f3", "e8e7", "f3g1", "e7e8", "g1f3", "e8e7", "f3g1"]:
        state = state.copy().make_move(chess.Move.from_uci(uci))
        assert_equivalent(state)
    assert state.is_over()
    for uci in ["e7e8", "a2a3", "e8e7", "g1f3", "e7e8", "f3g1"]:
        state = state.copy().make_move(chess.Move.from_uci(uci))
        assert_equivalent(state)
    assert not state.is_over()


def test_repetition_interrupted_by_castling_rights_loss():
    shuffle = ["g1f3", "e8e7", "f3g1", "e7e8"]
    board = chess.Board("4k3/8/8/8/8/8/8/R3K1N1 w Q - 0 1")
    for uci in shuffle + shuffle[:3]:
        board.push_uci(uci)
    assert ChessGameState(board).is_over()
    board = chess.Board("4k3/8/8/8/8/8/8/R3K1N1 w Q - 0 1")
    # Losing the castling right resets the count.
    for uci in shuffle + ["a1b1", "e8e7", "g1f3", "e7e8", "f3g1"]:
        board.push_uci(uci)
        assert_equivalent(ChessGameState(board))
    assert not ChessGameState(board).is_over()


def test_fifty_move_claim_and_seventyfive():
    for clock, over in [(97, False), (98, False), (99, True), (100, True), (150, True)]:
        state = ChessGameState(chess.Board(f"8/5k2/8/8/3K4/8/8/R7 w - - {clock} 120"))
        assert state.is_over() == over, clock
        assert_equivalent(state)
    state = ChessGameState(chess.Board("8/5k2/8/8/3K4/8/8/R7 w - - 150 120"))
    assert termination(state) == chess.Termination.SEVENTYFIVE_MOVES


def test_fivefold_repetition():
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 4:
        board.push_uci(uci)
    state = ChessGameState(board)
    assert termination(state) == chess.Termination.FIVEFOLD_REPETITION
    assert_equivalent(state)


def test_checkmate_stalemate_insufficient_material():
    mate = ChessGameState(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))
    assert mate.is_over() and mate.get_winner() is None
    mate = ChessGameState(chess.Board("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1"))
    assert mate.is_over() and mate.get_winner() == "white"
    stalemate = ChessGameState(chess.Board("7k/5Q2/8/6K1/8/8/8/8 b - - 0 1"))
    assert stalemate.is_over() and stalemate.get_winner() is None
    assert termination(stalemate) == chess.Termination.STALEMATE
    bare = ChessGameState(chess.Board("8/8/4k3/8/8/4K1N1/8/8 w - - 0 1"))
    assert bare.is_over() and termination(bare) == chess.Termination.INSUFFICIENT_MATERIAL
    for state in (mate, stalemate, bare):
        assert_equivalent(state)


def test_make_move_invalidates_the_cached_outcome_and_copy_is_independent():
    state = ChessGameState(chess.Board("8/5k2/8/8/3K4/8/8/R7 w - - 99 120"))
    assert state.is_over()
    child = state.copy().make_move(chess.Move.from_uci("a1a8"))
    assert child.is_over() == child.get_board().is_game_over(claim_draw=True)
    assert state.is_over() and state.get_board().halfmove_clock == 99
