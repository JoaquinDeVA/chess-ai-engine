"""Playing-strength check: a short match of an ML bot against reference bots."""

import random
from typing import Any, Dict, Tuple

import chess

import models  # noqa: F401  (registers the built-in models)
from bots.bot_registry import BotRegistry
from bots.chess_bot import ChessBot
from bots.heuristic_alphabeta_bot import HeuristicAlphaBetaBot
from bots.ml_bot import MLBot
from bots.random_bot import RandomBot
from engine.game_manager import GameManager
from evaluation.heuristics import h2, h3

OPENING_PLIES = 4


def random_opening(seed: int) -> str:
    """FEN reached after a few random plies, so that games do not all repeat.

    :param seed: Seed of the random choice.
    :return: FEN of the resulting position.
    """
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(OPENING_PLIES):
        board.push(rng.choice(list(board.legal_moves)))
    return board.fen()


def reference_opponents(depth: int) -> Dict[str, ChessBot]:
    """Opponents used for the strength column of ``MODELS_HISTORY.md``.

    :param depth: Search depth of the heuristic opponents (same as the ML bot's).
    :return: Mapping from display name to bot.
    """
    return {
        "RandomBot": RandomBot(),
        "H2": HeuristicAlphaBetaBot(h2, depth, name="H2"),
        "H3": HeuristicAlphaBetaBot(h3, depth, name="H3"),
    }


def run_strength_match(
    model_name: str,
    games: int = 10,
    depth: int = 2,
    time_limit: float = 180.0,
    increment: float = 2.0,
) -> Dict[str, Any]:
    """Play ``games`` games against each reference bot, alternating colours.

    Every pair of games starts from the same random opening with colours swapped.

    :param model_name: Registered model the ML bot evaluates with.
    :param games: Games per opponent.
    :param depth: Search depth of the ML bot and the heuristic opponents.
    :param time_limit: Seconds per player.
    :param increment: Seconds added per move.
    :return: Card section: depth, time control and ``W-D-L`` from the model's point of view.
    """
    manager = GameManager(BotRegistry())
    bot = MLBot(model=model_name, max_depth=depth)
    results: Dict[str, str] = {}

    for opponent_name, opponent in reference_opponents(depth).items():
        wins = draws = losses = 0
        for game in range(games):
            opening = random_opening(game // 2)
            players: Tuple[ChessBot, ChessBot] = (
                (bot, opponent) if game % 2 == 0 else (opponent, bot)
            )
            result = manager.play_game(
                players[0], players[1], time_limit, increment, starting_position=opening
            )
            if result.winner is None:
                draws += 1
            elif (result.winner == "white") == (players[0] is bot):
                wins += 1
            else:
                losses += 1
        results[opponent_name] = f"{wins}-{draws}-{losses}"

    return {
        "depth": depth,
        "time_control": f"{time_limit:g}+{increment:g}",
        "games_per_opponent": games,
        "results": results,
    }
