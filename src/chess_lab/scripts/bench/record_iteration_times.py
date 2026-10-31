#!/usr/bin/env python3
"""Record how long every iterative-deepening iteration takes, in the positions of real games.

python src/chess_lab/scripts/bench/record_iteration_times.py data/tournament/games_10min --player linear_material --games 2

The positions are the ones a player searched in games already played (the per-move logs written by
``play_tournament.py --logs``), searched again in order by a bot that keeps its table between moves
as in a game. The deepening is not stopped by the time estimate or the soft budget, only by the hard
limit, so every move gives the full series of iteration durations. ``compare_time_estimators.py``
replays the stopping rules on these series.

Writes one JSON line per move to ``--out`` (default ``data/reports/iteration_times.jsonl``).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import chess

from chess_engine.bots.time_manager import TimeManager
from chess_lab.players import make_player

#: Clock handed to the bot at every move: the budget is then the 40 s cap of the time manager.
CLOCK_SECONDS = 600.0
#: Deepest iteration tried; the hard limit ends the search long before in practice.
MAX_DEPTH = 16


def moves_of(game: Dict[str, Any], player: str) -> List[Dict[str, Any]]:
    """Moves of one player in a game log, in play order.

    :param game: Parsed per-game log (``white``, ``black`` and ``moves``).
    :param player: Player name.
    :return: The move records of that player.
    """
    color = "white" if game["white"] == player else "black" if game["black"] == player else None
    if color is None:
        return []
    return sorted((m for m in game["moves"] if m["color"] == color), key=lambda m: m["ply"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="folder with the per-game *.json logs")
    parser.add_argument("--player", required=True, help="player whose positions are replayed")
    parser.add_argument("--games", type=int, default=2, help="how many games of that player")
    parser.add_argument("--out", type=Path, default=Path("data/reports/iteration_times.jsonl"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Replay the games and write the series.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    games = []
    for path in sorted(args.log_dir.glob("*.json")):
        game = json.loads(path.read_text(encoding="utf-8"))
        moves = moves_of(game, args.player)
        if moves:
            games.append((path.name, moves))
    if not games:
        print(f"no games of {args.player} in {args.log_dir}")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as out:
        for name, moves in games[: args.games]:
            bot = make_player(args.player, TimeManager(), max_depth=MAX_DEPTH)
            bot.on_game_start(chess.WHITE, "recorder")
            bot.estimator = "never_stop"
            bot.min_depth = MAX_DEPTH  # only the hard limit may end an iteration
            for move in moves:
                bot.get_move(chess.Board(move["fen"]), CLOCK_SECONDS)
                stats = bot.last_search_stats
                record = {
                    "game": name,
                    "ply": move["ply"],
                    "player": args.player,
                    "budget": bot.turn_budget,
                    "iterations": [list(it) for it in stats.iterations],
                    "stop_reason": stats.stop_reason,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
            print(f"{name}: {len(moves)} moves recorded", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
