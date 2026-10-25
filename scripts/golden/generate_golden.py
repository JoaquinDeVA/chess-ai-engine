#!/usr/bin/env python3
"""Generate the golden files that pin the behaviour of the engine.

python scripts/golden/generate_golden.py            # everything
python scripts/golden/generate_golden.py --only positions,terminal

The files in ``tests/golden/`` are produced by the Python engine and are the executable
specification of the Rust core: positions, terminal tests, features, evaluations, fixed-depth
searches, opening-book lookups and tablebase moves. Everything is seeded, so running the script
twice writes the same files.

Ordering contract. Where the result depends on the order in which equal moves are tried, the
legal moves are ordered by their UCI string (``move_sort_key``), here and in the Rust core. The
production Python code orders them as python-chess generates them; the searches below use a
subclass that sorts them, which is the only difference.
"""

import argparse
import json
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import chess

from chess_engine.evaluation.features import get_position_features
from chess_engine.evaluation.heuristics import FEATURE_NAMES, h1
from chess_engine.knowledge.endgame_tablebase import EndgameTablebase
from chess_engine.knowledge.opening_book import OpeningBook
from chess_engine.models import load_model
from chess_engine.models.evaluator import ModelEvaluator
from chess_engine.search.chess_game_state import ChessGameState
from chess_engine.search.chess_problem import ChessProblem
from chess_engine.search.game_algorithms import advanced_heuristic_alphabeta_search
from chess_engine.search.search_utils import order_by_mvv_lva

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden"
SEED = 42
POSITIONS = 2000
MODELS = ("linear_material", "linear_pst", "linear_pst_terms", "linear_king_bucket", "net4_pst")
PLAYERS = ("h1", *MODELS)
#: Positions searched per player at depths 1 to 3 (with and without quiescence), and at depth 4.
SEARCH_POSITIONS, DEEP_POSITIONS = 40, 10
START_FEN = chess.STARTING_FEN


def move_sort_key(move: chess.Move) -> str:
    """The ordering contract: legal moves are tried in the order of their UCI string."""
    return move.uci()


class SortedChessProblem(ChessProblem):
    """Chess problem whose actions follow the ordering contract."""

    def actions(self, state):
        """Legal moves of ``state`` ordered by UCI."""
        return sorted(state.get_legal_moves(), key=move_sort_key)


def sorted_noisy_moves(state) -> List[chess.Move]:
    """Captures and promotions, ordered by the contract."""
    board = state.get_board()
    noisy = [m for m in board.legal_moves if m.promotion or board.is_capture(m)]
    return sorted(noisy, key=move_sort_key)


def write(name: str, payload: Any) -> None:
    """Write one golden file, compact and with sorted keys."""
    GOLDEN.mkdir(parents=True, exist_ok=True)
    path = GOLDEN / name
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e3:.0f} kB)")


def replay(entry: Dict[str, Any]) -> chess.Board:
    """The board of a position entry: its root and the moves played from it."""
    board = chess.Board(entry["root"])
    for uci in entry["moves"]:
        board.push_uci(uci)
    return board


def random_positions(count: int, seed: int) -> List[Dict[str, Any]]:
    """Positions of random games, each with the moves that lead to it."""
    rng = random.Random(seed)
    entries: List[Dict[str, Any]] = []
    while len(entries) < count:
        board = chess.Board()
        moves: List[str] = []
        length = rng.randint(0, 160)
        while len(moves) < length and not board.is_game_over(claim_draw=True):
            move = rng.choice(list(board.legal_moves))
            moves.append(move.uci())
            board.push(move)
        entries.append({"root": START_FEN, "moves": moves, "fen": board.fen()})
    return entries


def constructed_cases() -> List[Dict[str, Any]]:
    """Hand-made terminal situations: mates, stalemates, insufficient material, repetitions."""
    fens = [
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # stalemate
        "7k/8/6K1/8/8/8/8/8 w - - 0 1",  # bare kings
        "7k/8/6K1/8/8/8/8/N7 w - - 0 1",  # king and knight
        "7k/8/6K1/8/8/8/B7/8 w - - 0 1",  # king and bishop
        "7k/8/6K1/8/8/8/8/NN6 w - - 0 1",  # two knights
        "7k/6b1/6K1/8/8/8/B7/8 w - - 0 1",  # bishops on opposite colours
        "7k/7b/6K1/8/8/8/B7/8 w - - 0 1",  # bishops on the same colour
        "7k/8/6K1/8/8/8/8/R7 w - - 0 1",  # king and rook
        "7k/8/6K1/8/8/8/P7/8 w - - 0 1",  # king and pawn
        "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1",  # no mate
        "R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1",  # back-rank mate
        "7k/8/6K1/8/8/8/8/8 w - - 99 80",  # fifty-move clock below the claim
        "7k/8/6K1/8/8/8/8/R7 w - - 100 80",  # fifty-move claim
        "7k/8/6K1/8/8/8/8/R7 w - - 150 100",  # seventy-five move rule
        "R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 100 80",  # mate beats the clock
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    ]
    cases = [{"root": fen, "moves": [], "fen": fen} for fen in fens]
    # Threefold repetition by shuffling knights, checked after every move.
    for prefix, shuffle in (
        ("", "g1f3 g8f6 f3g1 f6g8"),
        ("", "b1c3 b8c6 c3b1 c6b8"),
        ("e2e4 e7e5", "g1f3 g8f6 f3g1 f6g8"),
    ):
        moves: List[str] = []
        board = chess.Board()
        for uci in prefix.split():
            moves.append(uci)
            board.push_uci(uci)
        for _ in range(3):
            for uci in shuffle.split():
                moves.append(uci)
                board.push_uci(uci)
                cases.append({"root": START_FEN, "moves": list(moves), "fen": board.fen()})
    # A position that repeats after an irreversible move, and a position with a legal en passant.
    for line in ("e2e4 e7e6 g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1", "e2e4 a7a6 e4e5 d7d5"):
        board = chess.Board()
        moves = []
        for uci in line.split():
            moves.append(uci)
            board.push_uci(uci)
            cases.append({"root": START_FEN, "moves": list(moves), "fen": board.fen()})
    return cases


def golden_positions() -> List[Dict[str, Any]]:
    """positions.json: random positions with their histories."""
    entries = random_positions(POSITIONS, SEED)
    write("positions.json", {"seed": SEED, "positions": entries})
    return entries


def golden_terminal(entries: List[Dict[str, Any]]) -> None:
    """terminal.json: whether the game is over and who won, as the search sees it."""
    out = []
    for entry in entries + constructed_cases():
        state = ChessGameState(replay(entry))
        winner = state.get_winner() if state.is_over() else None
        out.append({**entry, "over": state.is_over(), "winner": winner})
    write("terminal.json", {"cases": out})


def golden_features(entries: List[Dict[str, Any]]) -> None:
    """features_*.json: the inputs of every evaluation."""
    h1_rows = []
    for entry in entries:
        features = get_position_features(chess.Board(entry["fen"]))
        h1_rows.append([features[name] for name in FEATURE_NAMES])
    write("features_h1.json", {"names": list(FEATURE_NAMES), "rows": h1_rows})
    for name in MODELS:
        model = load_model(name)
        perspective = model.rep.perspective
        rows = []
        for entry in entries:
            indices, values = model.features.active(chess.Board(entry["fen"]), perspective)
            rows.append([indices, values])
        write(
            f"features_{name}.json",
            {"features": model.features.name, "perspective": perspective, "rows": rows},
        )


def golden_evaluations(entries: List[Dict[str, Any]]) -> None:
    """eval_*.json: the expected score of White and the evaluator value of each colour."""
    rows = []
    for entry in entries:
        state = ChessGameState(chess.Board(entry["fen"]))
        rows.append([h1(state, chess.WHITE), h1(state, chess.BLACK)])
    write("eval_h1.json", {"rows": rows})
    for name in MODELS:
        model = load_model(name)
        evaluator = ModelEvaluator(model)
        rows = []
        for entry in entries:
            board = chess.Board(entry["fen"])
            state = ChessGameState(board)
            rows.append(
                [
                    model.expected_score(board),
                    evaluator(state, chess.WHITE),
                    evaluator(state, chess.BLACK),
                ]
            )
        write(f"eval_{name}.json", {"rows": rows})


def player_eval(player: str):
    """The evaluation function ``(state, colour) -> value`` of a player."""
    if player == "h1":
        return h1
    return ModelEvaluator(load_model(player))


def search_player(player: str, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fixed-depth searches of one player: value, move and counters, with and without quiescence."""
    evaluate = player_eval(player)
    results = []
    for index, entry in enumerate(entries[:SEARCH_POSITIONS]):
        for depth in (1, 2, 3, 4):
            if depth == 4 and index >= DEEP_POSITIONS:
                continue
            for quiescence in (0, 4):
                if depth == 4 and quiescence:
                    continue
                board = replay(entry)
                state = ChessGameState(board)
                if state.is_over():
                    continue
                table: Dict[Any, Any] = {}
                move, stats = advanced_heuristic_alphabeta_search(
                    game=SortedChessProblem(),
                    state=state,
                    eval_fn=evaluate,
                    cutoff_test=lambda s, d, t: d == 0,
                    killer_moves={},
                    transposition_table=table,
                    order_moves_callback=order_by_mvv_lva,
                    max_depth=depth,
                    noisy_moves=sorted_noisy_moves if quiescence else None,
                    quiescence_plies=quiescence,
                )
                root = table[state.key()].result
                results.append(
                    {
                        "index": index,
                        "depth": depth,
                        "quiescence": quiescence,
                        "value": root.value,
                        "move": move.uci() if move else None,
                        "nodes": stats.nodes_visited,
                        "prunings": stats.pruning_count,
                        "max_ply": stats.max_depth_reached,
                    }
                )
    return {"player": player, "results": results}


def golden_search(entries: List[Dict[str, Any]]) -> None:
    """search_<player>.json: one process per player."""
    with ProcessPoolExecutor(max_workers=len(PLAYERS)) as pool:
        futures = [pool.submit(search_player, player, entries) for player in PLAYERS]
        for future in futures:
            payload = future.result()
            write(f"search_{payload['player']}.json", payload)


def golden_book() -> None:
    """book.json: lookups in the shipped book, hits and misses."""
    book = OpeningBook.shipped()
    rng = random.Random(SEED)
    keys = sorted(book._moves)
    hits = [{"fen": key, "move": book._moves[key]} for key in rng.sample(keys, 200)]
    misses = []
    for entry in random_positions(400, SEED + 1):
        board = chess.Board(entry["fen"])
        if book.lookup(board) is None:
            misses.append({"fen": entry["fen"], "move": None})
    write("book.json", {"size": len(book), "cases": hits + misses[:200]})


def tablebase_cases(directory: Path, pieces: List[str], count: int, seed: int) -> List[Dict]:
    """Random legal positions with the given extra pieces and the tablebase move for each."""
    rng = random.Random(seed)
    tables = EndgameTablebase(directory)
    cases: List[Dict[str, Any]] = []
    while len(cases) < count:
        extra = rng.choice(pieces)
        white_extra, black_extra = (extra.split("v") + [""])[:2]
        board = chess.Board(None)
        squares = rng.sample(range(64), 2 + len(white_extra) + len(black_extra))
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        cursor = 2
        for symbol, color in [(c, chess.WHITE) for c in white_extra] + [
            (c, chess.BLACK) for c in black_extra
        ]:
            board.set_piece_at(
                squares[cursor], chess.Piece(chess.Piece.from_symbol(symbol).piece_type, color)
            )
            cursor += 1
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        if not board.is_valid() or board.is_game_over():
            continue
        move = sorted_best_move(tables, board)
        cases.append({"fen": board.fen(), "move": move.uci() if move else None})
    tables.close()
    return cases


def sorted_best_move(tables: EndgameTablebase, board: chess.Board):
    """``EndgameTablebase.best_move`` with ties resolved by the ordering contract."""
    if not tables.covers(board) or board.is_game_over():
        return None
    best_move, best_key = None, None
    probe = board.copy(stack=False)
    for move in sorted(board.legal_moves, key=move_sort_key):
        probe.push(move)
        try:
            key = tables._rank(probe)
        except Exception:  # missing table
            return None
        finally:
            probe.pop()
        if best_key is None or key > best_key:
            best_move, best_key = move, key
    return best_move


def golden_tablebase() -> None:
    """tablebase.json: the tests' three-piece tables, and the full set when it is present."""
    small = tablebase_cases(
        ROOT / "tests" / "data" / "syzygy", ["Q", "R", "B", "N", "P", "QvR"], 150, SEED
    )
    payload: Dict[str, Any] = {"three_piece": small, "five_piece": []}
    full = ROOT / "data" / "knowledge" / "endgame"
    if full.is_dir():
        payload["five_piece"] = tablebase_cases(
            full, ["QR", "RR", "QvR", "RBvN", "PPvP", "QvRP", "RNvB", "QPvQ"], 150, SEED + 7
        )
    write("tablebase.json", payload)


def git_head() -> str:
    """The commit the golden files were generated from."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main(argv=None) -> int:
    """Generate the requested files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        help="comma-separated parts: positions,terminal,features,evaluations,search,book,tablebase",
    )
    args = parser.parse_args(argv)
    parts = set(args.only.split(",")) if args.only else None

    def wanted(part: str) -> bool:
        return parts is None or part in parts

    entries = (
        golden_positions()
        if wanted("positions")
        else json.loads((GOLDEN / "positions.json").read_text())["positions"]
    )
    if wanted("terminal"):
        golden_terminal(entries)
    if wanted("features"):
        golden_features(entries)
    if wanted("evaluations"):
        golden_evaluations(entries)
    if wanted("book"):
        golden_book()
    if wanted("tablebase"):
        golden_tablebase()
    if wanted("search"):
        golden_search(entries)
    (GOLDEN / "README.md").write_text(
        "# Golden files\n\nGenerated by `python scripts/golden/generate_golden.py` from the Python engine at "
        f"commit `{git_head()}`. They are the specification the Rust core must match; the tests "
        "that read them are `tests/test_core_*.py`.\n\nLegal moves are ordered by their UCI string wherever the order "
        "matters.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
