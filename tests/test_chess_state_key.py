"""``ChessGameState.key()`` must identify exactly the same positions as ``board.fen()``."""

import random

import chess

from chess_engine.search.chess_game_state import ChessGameState

START_FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    # En passant available at once (legal) and one that is pseudo-legal only (pinned pawn).
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "4k3/8/8/8/1pP5/8/8/4K2r b - c3 0 1",
    # Castling rights in the FEN that do not match the pieces (cleaned by python-chess).
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/4K2R w KQkq - 5 9",
    "4k3/8/8/8/8/8/8/R3K2R w Qq - 0 1",
]


def collect_positions(seed_count: int = 30, plies: int = 120) -> list[chess.Board]:
    """Positions of random games (pawn pushes and castling are favoured), as independent boards.

    :param seed_count: Number of games per start position.
    :param plies: Maximum plies per game.
    :return: Copies of every position reached.
    """
    positions = []
    for fen in START_FENS:
        for seed in range(seed_count):
            rng = random.Random(seed)
            board = chess.Board(fen)
            positions.append(board.copy())
            for _ in range(plies):
                moves = list(board.legal_moves)
                if not moves:
                    break
                special = [
                    m
                    for m in moves
                    if board.is_castling(m)
                    or board.is_en_passant(m)
                    or (board.piece_type_at(m.from_square) == chess.PAWN)
                ]
                move = rng.choice(special) if special and rng.random() < 0.5 else rng.choice(moves)
                board.push(move)
                positions.append(board.copy())
    return positions


def test_key_is_equivalent_to_the_fen():
    keys_by_fen: dict[str, object] = {}
    fens_by_key: dict[object, str] = {}
    en_passant = castling = 0
    for board in collect_positions():
        fen, key = board.fen(), ChessGameState(board).key()
        assert keys_by_fen.setdefault(fen, key) == key, fen
        assert fens_by_key.setdefault(key, fen) == fen, (fen, fens_by_key[key])
        en_passant += board.has_legal_en_passant()
        castling += bool(board.castling_rights)
    # The sample really contains the interesting cases and many distinct positions.
    assert en_passant > 20 and castling > 100
    assert len(keys_by_fen) > 3000


def test_key_ignores_the_move_stack_but_not_the_counters():
    played = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        played.push_uci(uci)
    fresh = chess.Board(played.fen())
    assert ChessGameState(played).key() == ChessGameState(fresh).key()
    other_clock = chess.Board(played.fen().replace(" 4 3", " 5 3"))
    assert ChessGameState(other_clock).key() != ChessGameState(played).key()
    assert ChessGameState(chess.Board()).key() != ChessGameState(played).key()


def test_apply_and_undo_move_restore_the_state_exactly():
    rng = random.Random(3)
    board = chess.Board("4k3/8/8/8/8/8/8/R3K1N1 w - - 0 1")
    state = ChessGameState(board)
    for _ in range(80):
        if state.is_over():
            break
        snapshot = (state.board.fen(), state._history, state._has_repeated, state._outcome)
        move = rng.choice(state.get_legal_moves())
        reference = state.copy().make_move(move)
        state.apply_move(move)
        assert state.key() == reference.key()
        assert state._history == reference._history
        assert state.is_over() == reference.is_over()
        state.undo_move()
        assert (state.board.fen(), state._history, state._has_repeated, state._outcome) == snapshot
        state.apply_move(move)  # keep walking
    assert len(state._undo_stack) > 0
