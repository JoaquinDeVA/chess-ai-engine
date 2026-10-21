#!/usr/bin/env python3
"""Summarize the search depth reached per move in the per-move logs of a tournament.

python scripts/bench/depth_in_game_logs.py data/tournament/reference

The logs are the ``*.json`` files that ``play_tournament.py --logs`` writes (one per game, with the
``completed_depth`` and the seconds of every move). For each player the script prints the number of
moves, the median and mean depth, the share of moves that finished below depth 3 and below depth 4,
and how many of the moves below depth 3 took less than 2 seconds (a search that stopped early
instead of running out of time).
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


def load_moves(log_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Read every game log of a folder and group the moves by player.

    :param log_dir: Folder with the per-game ``*.json`` logs.
    :return: ``{player: [move records]}``; moves without a completed depth are skipped.
    """
    moves: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in sorted(log_dir.glob("*.json")):
        game = json.loads(path.read_text(encoding="utf-8"))
        for move in game["moves"]:
            if move.get("completed_depth") is None:
                continue
            player = game["white"] if move["color"] == "white" else game["black"]
            moves[player].append(move)
    return dict(moves)


def summarize(moves: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Depth statistics of the moves of one player.

    :param moves: Move records with ``completed_depth`` and ``move_seconds``.
    :return: Count, median and mean depth, shares below depth 3 and 4, and the number of
        moves below depth 3 that took under 2 seconds.
    :raises ValueError: If there are no moves.
    """
    if not moves:
        raise ValueError("no moves to summarize")
    depths = [m["completed_depth"] for m in moves]
    below3 = [m for m in moves if m["completed_depth"] < 3]
    return {
        "moves": len(moves),
        "median": statistics.median(depths),
        "mean": statistics.fmean(depths),
        "below3": len(below3) / len(moves),
        "below4": sum(d < 4 for d in depths) / len(moves),
        "below3_fast": sum(m["move_seconds"] < 2.0 for m in below3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print the summary of a log folder.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="folder with the per-game *.json logs")
    args = parser.parse_args(argv)
    moves = load_moves(args.log_dir)
    if not moves:
        print(f"no move logs in {args.log_dir}")
        return 1
    print("| player | moves | median depth | mean depth | below 3 | below 4 | below 3 in < 2 s |")
    print("|---|---|---|---|---|---|---|")
    for player, records in sorted(moves.items()):
        s = summarize(records)
        print(
            f"| {player} | {s['moves']} | {s['median']:g} | {s['mean']:.2f} | "
            f"{s['below3']:.1%} | {s['below4']:.1%} | {s['below3_fast']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
