#!/usr/bin/env python3
"""
Chess AI Engine - command-line interface.

Play a game between any two players (humans or bots), or run a round-robin tournament between
all the bots. The CLI skeleton comes from the AI course framework (CEU San Pablo University);
``--model`` and ``--depth`` are additions.
"""

import argparse
import inspect
import logging
import random
import sys
import traceback
from typing import Optional

from chess_engine import _core
from chess_engine.bots.bot_registry import BotRegistry
from chess_engine.bots.human_player import HumanPlayer
from chess_engine.config.settings import load_config
from chess_engine.engine.game_manager import GameManager
from chess_engine.knowledge.endgame_tablebase import EndgameTablebase
from chess_engine.knowledge.opening_book import OpeningBook


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up command line argument parsing.

    :return: Configured argument parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Chess Engine - Play against bots and run tournaments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --player1 human --player2 IterativeHeuristicBot          # Human(White) vs bot(Black)
  %(prog)s --player1 IterativeHeuristicBot --player2 HumanPlayer    # bot(White) vs Human(Black)
  %(prog)s --player1 IterativeHeuristicBot --random-color           # Human vs bot (random colors)
  %(prog)s --player2 IterativeHeuristicBot                          # Human vs bot (human=white)
  %(prog)s --player1 IterativeModelBookBot --model cnn_resnet --player2 IterativeHeuristicBot --depth 2
  %(prog)s --mode tournament
  %(prog)s --list-bots
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["single", "tournament"],
        default="single",
        help="Game mode: single (Player vs Player) or tournament (all bots) (default: %(default)s)",
    )

    parser.add_argument(
        "--player1",
        type=str,
        help="First player (bot name or 'human'/'HumanPlayer'). Defaults to 'human' if not specified.",
    )

    parser.add_argument(
        "--player2",
        type=str,
        help="Second player (bot name or 'human'/'HumanPlayer'). Defaults to 'human' if not specified.",
    )

    parser.add_argument(
        "--random-color",
        action="store_true",
        help="Assign colors randomly. By default player1=white, player2=black.",
    )

    parser.add_argument(
        "--position", type=str, help="Starting position as a FEN string (6 fields, in quotes)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Evaluation model for the machine-learning bots in single games; ignored in tournament mode",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Search depth (maximum depth for the iterative-deepening bots)",
    )

    parser.add_argument(
        "--opening-book",
        default=None,
        metavar="FILE",
        help="Opening book to play from (default data/opening_book.json, on for every search bot)",
    )
    parser.add_argument(
        "--no-opening-book", action="store_true", help="Search from the first move: no opening book"
    )
    parser.add_argument(
        "--endgame-db",
        default=None,
        metavar="DIR",
        help="Syzygy endgame tables to play from (default data/knowledge/endgame, on for every "
        "search bot when the folder exists)",
    )
    parser.add_argument(
        "--no-endgame-db",
        action="store_true",
        help="Search every endgame position: no endgame tables",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help="Threads that search (default 1); the search is then not deterministic",
    )
    parser.add_argument(
        "--techniques",
        default=None,
        metavar="LIST",
        help="Techniques of the search, comma-separated, 'all' or 'none' ("
        + ", ".join(_core.TECHNIQUES)
        + "); default: all but lmr",
    )
    parser.add_argument(
        "--hash",
        type=int,
        default=None,
        metavar="MB",
        help="Size in MB of the transposition table shared by several threads (default 64)",
    )

    parser.add_argument("--list-bots", action="store_true", help="List all available bots and exit")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--board-style",
        choices=["ascii", "emoji"],
        default="emoji",
        help="Board visualization style: ascii or emoji pieces (default: %(default)s)",
    )

    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export games to PGN. For single games: specify .pgn file (default: last_game.pgn). For tournaments: specify directory (default: last_tournament/)",
    )

    return parser


def list_available_bots(bot_registry: BotRegistry) -> None:
    """
    List all available bots in the registry.

    :param bot_registry: The bot registry instance
    :type bot_registry: BotRegistry
    """
    print("Available Chess Bots:")
    print("=" * 40)

    available_bots = bot_registry.list_available_bots()

    if not available_bots:
        print("No bots found in the bots/ directory.")
        print("Make sure you have implemented bot classes that inherit from ChessBot.")
        return

    for i, bot_name in enumerate(available_bots, 1):
        print(f"{i:2d}. {bot_name}")

        bot_info = bot_registry.get_bot_info(bot_name)
        if bot_info and "author" in bot_info:
            print(f"     Author: {bot_info['author']}")

    print(f"\nTotal: {len(available_bots)} bots available")


def load_position_from_argument(position_arg: str) -> Optional[str]:
    """
    Load a chess position from command line argument.

    :param position_arg: A FEN string
    :type position_arg: str
    :return: Valid FEN string, or None if invalid
    :rtype: Optional[str]
    """
    # A FEN string has 6 parts separated by spaces
    parts = position_arg.split()
    if len(parts) == 6:
        # Basic validation: the placement field may only contain pieces, digits and slashes
        valid_chars = set("rnbqkpRNBQKP12345678/")
        if all(c in valid_chars for c in parts[0]):
            return position_arg

    return None


def bot_options(bot_registry: BotRegistry, bot_name: str, args: argparse.Namespace) -> dict:
    """
    Build the constructor options requested on the command line for a bot.

    ``--model``, ``--depth``, the endgame tables and the opening book are only passed to bots whose
    constructor accepts them (``model``, ``max_depth``, ``tablebase`` and ``opening_book``), so they
    can be given freely in a mixed match. The book and the tables are on unless switched off.

    :param bot_registry: The bot registry instance
    :type bot_registry: BotRegistry
    :param bot_name: Name of the bot
    :type bot_name: str
    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    :return: Keyword arguments for :meth:`BotRegistry.get_bot`
    :rtype: dict
    """
    parameters = inspect.signature(bot_registry.bot_classes[bot_name].__init__).parameters
    options = {}
    if args.model is not None and "model" in parameters:
        options["model"] = args.model
    if args.depth is not None and "max_depth" in parameters:
        options["max_depth"] = args.depth
    takes_search_options = "search_options" in parameters
    if takes_search_options or "threads" in parameters:
        if args.threads is not None:
            options["threads"] = args.threads
        if args.hash is not None:
            options["hash_mb"] = args.hash
    if args.techniques is not None and (takes_search_options or "techniques" in parameters):
        names = {"all": list(_core.TECHNIQUES), "none": []}.get(
            args.techniques, args.techniques.split(",")
        )
        options["techniques"] = tuple(n for n in names if n)
    if "tablebase" in parameters or takes_search_options:
        if args.no_endgame_db:
            options["tablebase"] = None
        elif args.endgame_db is not None:
            options["tablebase"] = EndgameTablebase(args.endgame_db)
    if "opening_book" in parameters or takes_search_options:
        if args.no_opening_book:
            options["opening_book"] = None
        elif args.opening_book is not None:
            options["opening_book"] = OpeningBook.load(args.opening_book)
    return options


def run_single_game(
    args: argparse.Namespace, game_manager: GameManager, bot_registry: BotRegistry
) -> bool:
    """
    Run a single chess game between two players.

    Supports any combination of human and bot players.
    If no players specified, defaults to human vs human.

    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    :param game_manager: Game manager instance
    :type game_manager: GameManager
    :param bot_registry: Bot registry instance
    :type bot_registry: BotRegistry
    :return: True if game completed successfully, False otherwise
    :rtype: bool
    """
    # Get player names (default to human if not specified)
    player1_name = args.player1 or "human"
    player2_name = args.player2 or "human"

    # Helper function to check if a player name refers to a human
    def is_human_player(name: str) -> bool:
        return name.lower() in ["human", "humanplayer", "human_player"]

    # Create players
    # Player 1
    if is_human_player(player1_name):
        player1 = HumanPlayer("Human (Player 1)")
    else:
        if not bot_registry.is_bot_available(player1_name):
            print(f"Error: Bot '{player1_name}' not found")
            print("Use --list-bots to see available bots")
            return False
        player1 = bot_registry.get_bot(
            player1_name, **bot_options(bot_registry, player1_name, args)
        )
        if player1 is None:
            print(f"Error: Bot '{player1_name}' could not be loaded (see the log above)")
            return False

    # Player 2
    if is_human_player(player2_name):
        player2 = HumanPlayer("Human (Player 2)")
    else:
        if not bot_registry.is_bot_available(player2_name):
            print(f"Error: Bot '{player2_name}' not found")
            print("Use --list-bots to see available bots")
            return False
        player2 = bot_registry.get_bot(
            player2_name, **bot_options(bot_registry, player2_name, args)
        )
        if player2 is None:
            print(f"Error: Bot '{player2_name}' could not be loaded (see the log above)")
            return False

    # Assign colors
    if args.random_color:
        if random.choice([True, False]):
            white_player, black_player = player1, player2
        else:
            white_player, black_player = player2, player1
    else:
        # Default: player1=white, player2=black
        white_player, black_player = player1, player2

    # Handle custom starting position
    starting_position = None
    if args.position:
        starting_position = load_position_from_argument(args.position)
        if not starting_position:
            print(f"Error: Invalid position '{args.position}'")
            print("Position should be a FEN string")
            return False

        if args.verbose:
            print(f"Starting from custom position: {starting_position}")

    # Load time controls from config
    try:
        config = load_config()
        time_limit = config.total_game_time
        time_increment = config.time_increment
    except Exception as e:
        if args.verbose:
            print(f"Warning: Could not load config, using defaults: {e}")
        time_limit = 60.0
        time_increment = 0.0

    # Disable time control if at least one player is human
    if isinstance(white_player, HumanPlayer) or isinstance(black_player, HumanPlayer):
        time_limit = float("inf")
        time_increment = 0.0
        if args.verbose:
            print("Human player detected - time control disabled")

    # Start the game
    print(f"Starting game: {white_player.name} (White) vs {black_player.name} (Black)")
    if starting_position:
        print("Custom starting position loaded")
    if args.verbose:
        print(f"Time control: {time_limit}s + {time_increment}s increment")

    try:
        game_result = game_manager.play_game(
            white_player=white_player,
            black_player=black_player,
            time_limit=time_limit,
            time_increment=time_increment,
            starting_position=starting_position,
            verbose=args.verbose,
            board_style=args.board_style,
        )

        # Display results
        print("\nGame Complete!")
        print("-" * 30)
        print(f"Result: {game_result.result}")
        print(f"Winner: {game_result.winner if game_result.winner else 'Draw'}")
        print(f"Total moves: {game_result.total_moves}")
        print(f"Game duration: {game_result.duration:.1f} seconds")
        print(f"White time left: {game_result.white_time_left:.1f}s")
        print(f"Black time left: {game_result.black_time_left:.1f}s")

        if args.verbose:
            print(f"Termination reason: {game_result.termination}")

        # Export to PGN
        export_path = args.export if args.export else "last_game.pgn"
        try:
            pgn_content = game_result.to_pgn(time_limit=time_limit, time_increment=time_increment)
            with open(export_path, "w") as f:
                f.write(pgn_content)
            print(f"Game exported to: {export_path}")
        except Exception as e:
            print(f"Error exporting game: {e}")
            if args.verbose:
                traceback.print_exc()

        return True

    except KeyboardInterrupt:
        print("\nGame cancelled by user")
        return False
    except Exception as e:
        print(f"Error during game: {e}")
        if args.verbose:
            traceback.print_exc()
        return False


def run_tournament(
    args: argparse.Namespace, game_manager: GameManager, bot_registry: BotRegistry
) -> bool:
    """
    Run tournament with all available bots.

    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    :param game_manager: Game manager instance
    :type game_manager: GameManager
    :param bot_registry: Bot registry instance
    :type bot_registry: BotRegistry
    :return: True if tournament completed successfully, False otherwise
    :rtype: bool
    """
    try:
        config = load_config()
        if args.verbose:
            print("Loaded tournament config from default location")
            print(f"Tournament rounds: {config.rounds}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("The tournament settings come from chess_engine.config.settings.load_config()")
        return False

    available_bots = bot_registry.get_tournament_ready_bots()

    if len(available_bots) < 2:
        print("Error: Tournament requires at least 2 bots")
        print(
            f"Found {len(available_bots)} tournament-ready bots. Use --list-bots to see all available bots."
        )
        print("Make sure you have implemented multiple bot classes.")
        return False

    print(f"Starting tournament with {len(available_bots)} bots")
    print(f"Bots: {', '.join(available_bots)}")
    print(f"Format: {config.rounds} round(s), round-robin")
    print("-" * 50)

    try:
        tournament_result = game_manager.run_tournament(
            bot_names=available_bots, config=config, verbose=args.verbose
        )

        print("\nTournament Complete!")
        print("=" * 50)

        # Show final standings
        standings = tournament_result.get_standings()
        print("\nFinal Standings:")
        print("-" * 40)
        print(f"{'Rank':<4} {'Bot':<15} {'Score':<6} {'W-D-L':<8} {'Games'}")
        print("-" * 40)

        for rank, (bot_name, stats) in enumerate(standings, 1):
            wdl = f"{stats['wins']}-{stats['draws']}-{stats['losses']}"
            games_played = stats["wins"] + stats["draws"] + stats["losses"]
            print(f"{rank:<4} {bot_name:<15} {stats['score']:<6.1f} {wdl:<8} {games_played}")

        print(f"\nTotal games played: {len(tournament_result.games)}")
        print(f"Tournament duration: {tournament_result.duration:.1f} seconds")

        safe_timestamp = tournament_result.timestamp.replace(":", "-")
        results_file = f"tournament_results_{safe_timestamp}.json"
        tournament_result.save_to_file(results_file)
        print(f"Detailed results saved to: {results_file}")

        # Export tournament games to PGN
        export_path = args.export if args.export else "last_tournament/"
        try:
            tournament_result.export_games_to_pgn_directory(export_path)
            print(f"Tournament games exported to directory: {export_path}")
            print(f"Exported {len(tournament_result.games)} PGN files")
        except Exception as e:
            print(f"Error exporting tournament games: {e}")
            if args.verbose:
                traceback.print_exc()

        return True

    except KeyboardInterrupt:
        print("\nTournament cancelled by user")
        return False
    except Exception as e:
        print(f"Error during tournament: {e}")
        if args.verbose:
            traceback.print_exc()
        return False


def main() -> int:
    """
    Main entry point for the chess engine.

    :return: Exit code (0 for success, 1 for error)
    :rtype: int
    """
    parser = setup_argument_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s"
    )

    if args.list_bots:
        try:
            bot_registry = BotRegistry()
            list_available_bots(bot_registry)
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1

    try:
        if args.verbose:
            print("Initializing chess engine...")

        bot_registry = BotRegistry()
        game_manager = GameManager(bot_registry)

        if args.verbose:
            available_count = len(bot_registry.list_available_bots())
            print(f"Found {available_count} available bots")

    except Exception as e:
        print(f"Error initializing chess engine: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

    success = False
    try:
        if args.mode == "single":
            success = run_single_game(args, game_manager, bot_registry)

        elif args.mode == "tournament":
            success = run_tournament(args, game_manager, bot_registry)

        else:
            print(f"Error: Unknown mode '{args.mode}'")
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
