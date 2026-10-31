#!/usr/bin/env python3
"""Measure what each search technique saves, by switching it off one at a time.

python src/chess_lab/scripts/bench/compare_search_features.py --player linear_material --positions 60 --depth 4

Every position of a fixed sample of the test split (seed 42) is searched at a fixed depth with
all techniques on (the baseline), with each one off, and with all off. For each variant the script
reports the nodes and the seconds per move, the share of moves equal to the baseline's, and the
share equal to the Stockfish move. The transposition table, the killer moves and the move ordering
only change the cost, never the result (up to ties); the quiescence search changes the result.

Writes ``data/reports/SEARCH_FEATURES.md``. Nothing is written to the model cards.
"""

import argparse
import contextlib
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Sequence, Tuple

import chess

from chess_engine.bots.time_manager import TimeManager
from chess_lab.comparison.evaluate import SAMPLE_SEED, TEST, TEST_SAMPLE_ROWS
from chess_lab.comparison.tradeoff import make_player, validate_specs
from chess_lab.training.sidecar import load_sample

#: Clock given to the bot, in seconds: large enough for the depth limit to end every search.
GENEROUS_CLOCK = 1e6
#: Variants in report order: name -> techniques that are switched off.
VARIANTS: Dict[str, Tuple[str, ...]] = {
    "baseline": (),
    "no transposition table": ("table",),
    "no killer moves": ("killers",),
    "no move ordering": ("ordering",),
    "no quiescence": ("quiescence",),
    "none of them": ("table", "killers", "ordering", "quiescence"),
}


#: Techniques of selective search measured by switching them on: name -> techniques on.
TECHNIQUE_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "with tt_move": ("tt_move",),
    "with history": ("history",),
    "with pvs": ("pvs",),
    "with aspiration": ("aspiration",),
    "with null_move": ("null_move",),
    "with lmr": ("lmr",),
    "with all of them": ("tt_move", "history", "pvs", "aspiration", "null_move", "lmr"),
}


@contextlib.contextmanager
def switched_off(techniques: Sequence[str], player, enabled: Sequence[str] = ()) -> Iterator[None]:
    """Switch techniques off, and techniques of selective search on, for the duration of the block.

    :param techniques: Any of ``"table"``, ``"killers"``, ``"ordering"``, ``"quiescence"``.
    :param player: Bot whose search settings are changed (and restored).
    :param enabled: Techniques of selective search to switch on (see ``TECHNIQUE_VARIANTS``).
    """
    saved = player.disabled, player.quiescence_plies, getattr(player, "techniques", ())
    player.disabled = tuple(name for name in techniques if name != "quiescence")
    player.techniques = tuple(enabled)
    if "quiescence" in techniques:
        player.quiescence_plies = 0
    try:
        yield
    finally:
        player.disabled, player.quiescence_plies, player.techniques = saved


def measure(
    spec: str,
    rows: List[Tuple[str, str]],
    depth: int,
    techniques: Sequence[str],
    enabled: Sequence[str] = (),
) -> Tuple[List[str], float, float]:
    """Search every position once with some techniques off.

    :param spec: Player specification (see :func:`make_player`).
    :param rows: ``(fen, stockfish move in UCI)`` pairs.
    :param depth: Maximum search depth.
    :param techniques: Techniques switched off.
    :param enabled: Techniques of selective search switched on.
    :return: ``(chosen moves in UCI, nodes per move, seconds per move)``.
    """
    player = make_player(spec, TimeManager(), max_depth=depth)
    moves: List[str] = []
    nodes = 0
    seconds = 0.0
    with switched_off(techniques, player, enabled):
        for fen, _ in rows:
            player.on_game_start(chess.WHITE, "profiler")  # fresh table per position
            started = time.perf_counter()
            move = player.get_move(chess.Board(fen), GENEROUS_CLOCK)
            seconds += time.perf_counter() - started
            moves.append(move.uci())
            nodes += player.last_search_stats.nodes_visited
    return moves, nodes / len(rows), seconds / len(rows)


def share_equal(first: Sequence[str], second: Sequence[str]) -> float:
    """Share of positions where two lists of moves agree.

    :param first: Moves of one variant.
    :param second: Moves of the other, same positions and order.
    :return: Value in ``[0, 1]``.
    :raises ValueError: If the lists have different lengths or are empty.
    """
    if len(first) != len(second) or not first:
        raise ValueError("both lists must cover the same, non-empty positions")
    return sum(a == b for a, b in zip(first, second, strict=True)) / len(first)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", default="linear_material", help="evaluation to search with")
    parser.add_argument("--positions", type=int, default=60)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--variants",
        help="comma-separated names of the variants to run (default: all); the baseline always runs",
    )
    parser.add_argument("--raw", type=Path, default=Path("data/lichess/raw"))
    parser.add_argument("--sidecar", type=Path, default=Path("data/lichess/sidecar"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/SEARCH_FEATURES.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, measure_fn: Callable = measure) -> int:
    """Run the comparison and write the report.

    :param argv: Command-line arguments.
    :param measure_fn: Replaced in tests.
    :return: Process exit code.
    """
    args = parse_args(argv)
    validate_specs([args.player])
    sample = load_sample(args.raw, args.sidecar, TEST, TEST_SAMPLE_ROWS, seed=SAMPLE_SEED)
    rows = [
        (fen, best)
        for fen, best in zip(sample["fen"], sample["best_move"], strict=True)
        if isinstance(best, str)
    ][: args.positions]
    stockfish = [best for _, best in rows]

    lines = [
        "# Search techniques switched off, or on, one at a time",
        "",
        f"`{args.player}` at depth {args.depth} on {len(rows)} test positions (seed {SAMPLE_SEED}). "
        "'Same as baseline' is the share of moves equal to the baseline's; "
        "'Stockfish' the share equal to the Stockfish move.",
        "",
        "| variant | nodes/move | s/move | same as baseline | Stockfish |",
        "|---|---|---|---|---|",
    ]
    wanted = None if args.variants is None else {v.strip() for v in args.variants.split(",")}
    every = {
        **{n: (t, ()) for n, t in VARIANTS.items()},
        **{n: ((), t) for n, t in TECHNIQUE_VARIANTS.items()},
    }
    unknown = (wanted or set()) - set(every)
    if unknown:
        raise SystemExit(f"unknown variants: {sorted(unknown)}; choose from {list(every)}")
    baseline_moves: List[str] = []
    for name, (techniques, enabled) in every.items():
        if wanted is not None and name != "baseline" and name not in wanted:
            continue
        if wanted is None and enabled:
            continue  # the techniques of selective search are measured when asked for
        moves, nodes, seconds = measure_fn(args.player, rows, args.depth, techniques, enabled)
        if not baseline_moves:
            baseline_moves = moves
        lines.append(
            f"| {name} | {nodes:.0f} | {seconds:.3f} | {share_equal(moves, baseline_moves):.3f} | "
            f"{share_equal(moves, stockfish):.3f} |"
        )
        print(lines[-1], flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
