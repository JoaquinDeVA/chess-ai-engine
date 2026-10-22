#!/usr/bin/env python3
"""Fixed-depth search benchmark on a fixed set of positions.

Prints, per position and bot, the time, nodes visited, nodes per second, deepest ply reached and
the chosen move. Time varies between runs, but the chosen move and the node count are
deterministic, so two runs (for example before and after an optimisation) can be compared:

    python scripts/bench/bench_search_speed.py --json docs/benchmarks/search_baseline.json
    python scripts/bench/bench_search_speed.py --compare docs/benchmarks/search_baseline.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import chess

from chess_engine.bots.iterative_heuristic_bot import IterativeHeuristicBot
from chess_engine.bots.iterative_model_book_bot import IterativeModelBookBot
from chess_engine.evaluation.heuristics import h1

#: Large enough for the time manager to give every move its maximum budget (40 s).
CLOCK = 10_000.0

START = chess.STARTING_FEN

#: ``(name, start FEN, UCI moves pushed on it, search must meet a threefold claim)``.
POSITIONS: list[tuple[str, str, list[str], bool]] = [
    ("opening", START, [], False),
    (
        "middlegame",
        "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 2 7",
        [],
        False,
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        [],
        False,
    ),
    ("endgame", "8/5pk1/6p1/8/3R4/6P1/5PK1/3r4 w - - 0 40", [], False),
    # Start position seen twice; Ng1 then Ng8 would repeat it a third time (claim on next move).
    (
        "repetition_claim_next_move",
        START,
        ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"],
        True,
    ),
    # Kings and rook shuffling: several positions already seen twice and no irreversible move since.
    (
        "repetition_endgame",
        "8/8/4k3/8/8/4K3/8/R7 w - - 0 1",
        ["e3d3", "e6d6", "d3e3", "d6e6", "e3d3", "e6d6"],
        True,
    ),
    # Rook endgame near the fifty-move claim: the clock reaches 99 inside the tree.
    ("fifty_move_clock", "8/5k2/8/8/3K4/8/8/R7 w - - 97 120", [], False),
]


def build_board(fen: str, moves: list[str]) -> chess.Board:
    """Board from a start FEN with the given UCI moves pushed (so the move stack is real).

    :param fen: Start position.
    :param moves: UCI moves to play from it.
    :return: The resulting board.
    """
    board = chess.Board(fen)
    for uci in moves:
        board.push_uci(uci)
    return board


def reaches_threefold_claim(board: chess.Board, depth: int) -> bool:
    """Whether some node within ``depth`` plies is a draw only because of a threefold claim.

    :param board: Root position (it is restored before returning).
    :param depth: Plies to explore.
    :return: True if such a node exists.
    """
    if board.outcome(claim_draw=False) is not None:
        return False
    if board.can_claim_threefold_repetition():
        return True
    if depth == 0:
        return False
    for move in list(board.legal_moves):
        board.push(move)
        try:
            if reaches_threefold_claim(board, depth - 1):
                return True
        finally:
            board.pop()
    return False


def bot_factories(depth: int) -> dict[str, Callable[[], Any]]:
    """Constructors of the benchmarked bots (a fresh bot is used per position).

    :param depth: Search depth.
    :return: Mapping bot label -> constructor.
    """
    return {
        "heuristic_h1": lambda: IterativeHeuristicBot(
            h1, max_depth=depth, opening_book=None, tablebase=None
        ),
        "model_mlp_features": lambda: IterativeModelBookBot(
            model="mlp_features", opening_book=None, tablebase=None, max_depth=depth
        ),
    }


def run(depth: int) -> list[dict[str, Any]]:
    """Search every position with every bot.

    :param depth: Fixed search depth.
    :return: One result dictionary per (position, bot).
    :raises RuntimeError: If a search was truncated (results would not be deterministic).
    """
    results = []
    for name, fen, moves, needs_claim in POSITIONS:
        board = build_board(fen, moves)
        if needs_claim and not reaches_threefold_claim(board, depth):
            raise RuntimeError(f"{name}: the search tree never reaches a threefold claim")
        for label, make_bot in bot_factories(depth).items():
            bot = make_bot()
            start = time.perf_counter()
            move = bot.get_move(board.copy(), CLOCK)
            elapsed = time.perf_counter() - start
            stats = bot.last_search_stats
            if stats is None or stats.truncated:
                raise RuntimeError(f"{name}/{label}: search truncated or missing")
            results.append(
                {
                    "position": name,
                    "bot": label,
                    "time": elapsed,
                    "nodes": bot.total_nodes_visited,
                    "nodes_per_second": bot.total_nodes_visited / elapsed,
                    "max_depth": stats.max_depth_reached,
                    "move": move.uci(),
                }
            )
    return results


def print_table(results: list[dict[str, Any]]) -> None:
    """Print the results and the totals per bot.

    :param results: Output of :func:`run`.
    """
    header = (
        f"{'position':28} {'bot':20} {'time(s)':>8} {'nodes':>9} {'nodes/s':>9} {'depth':>5} move"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['position']:28} {r['bot']:20} {r['time']:8.2f} {r['nodes']:9d} "
            f"{r['nodes_per_second']:9.0f} {r['max_depth']:5d} {r['move']}"
        )
    print("-" * len(header))
    for label in dict.fromkeys(r["bot"] for r in results):
        rows = [r for r in results if r["bot"] == label]
        total_time = sum(r["time"] for r in rows)
        total_nodes = sum(r["nodes"] for r in rows)
        print(
            f"{'TOTAL':28} {label:20} {total_time:8.2f} {total_nodes:9d} "
            f"{total_nodes / total_time:9.0f}"
        )


def compare(results: list[dict[str, Any]], old_path: Path) -> bool:
    """Report differences in chosen move or node count against a saved run.

    :param results: Current results.
    :param old_path: JSON written by a previous run with ``--json``.
    :return: True if every position and bot matches.
    """
    old = json.loads(old_path.read_text())["results"]
    old_by_key = {(r["position"], r["bot"]): r for r in old}
    same = True
    for r in results:
        before = old_by_key.get((r["position"], r["bot"]))
        if before is None:
            print(f"DIFF {r['position']}/{r['bot']}: not in {old_path}")
            same = False
            continue
        for field in ("move", "nodes", "max_depth"):
            if before[field] != r[field]:
                print(f"DIFF {r['position']}/{r['bot']}: {field} {before[field]} -> {r[field]}")
                same = False
    if len(old_by_key) != len(results):
        print(f"DIFF: {len(old_by_key)} saved results vs {len(results)} current ones")
        same = False
    old_time = sum(r["time"] for r in old)
    new_time = sum(r["time"] for r in results)
    print(f"total time {old_time:.2f}s -> {new_time:.2f}s (x{old_time / new_time:.2f})")
    print("compare: identical moves and node counts" if same else "compare: DIFFERENCES FOUND")
    return same


def main(argv: Optional[list[str]] = None) -> int:
    """Run the benchmark.

    :param argv: Command-line arguments (``sys.argv[1:]`` by default).
    :return: Process exit code (1 if ``--compare`` found differences).
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--depth", type=int, default=3, help="fixed search depth")
    parser.add_argument("--json", type=Path, help="write the results to this JSON file")
    parser.add_argument("--compare", type=Path, help="compare against a JSON saved with --json")
    args = parser.parse_args(argv)

    results = run(args.depth)
    print_table(results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"depth": args.depth, "results": results}, indent=2) + "\n")
    if args.compare and not compare(results, args.compare):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
