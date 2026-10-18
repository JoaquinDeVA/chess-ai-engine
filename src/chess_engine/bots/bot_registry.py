"""
Bot registry: automatic bot discovery and loading.

Discovers every :class:`~bots.chess_bot.ChessBot` subclass defined in a public module of the
``bots/`` directory, so adding a bot is just adding a file (see ``docs/ARCHITECTURE.md``).
Modules whose name starts with an underscore are ignored.

Originally part of the AI course framework (CEU San Pablo University); extended with
``get_bot(name, **options)``, logging instead of printing, and the underscore convention.
"""

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

# Import the base bot class
from .chess_bot import ChessBot

logger = logging.getLogger(__name__)


class BotRegistry:
    """
    Registry for discovering and managing chess bots.

    This class automatically scans the bots/ directory for Python files
    containing ChessBot implementations and makes them available for games.
    """

    def __init__(self, bots_directory: Optional[str] = None):
        """
        Initialize the bot registry.

        :param bots_directory: Directory to scan for bot files; the ``bots`` package directory
            by default, so the registry works from any working directory
        :type bots_directory: Optional[str]
        """
        self.bots_directory = (
            Path(bots_directory) if bots_directory else Path(__file__).resolve().parent
        )
        self.loaded_bots: Dict[str, ChessBot] = {}
        self.available_bots: Dict[str, str] = {}  # bot_name -> file_path
        self.bot_classes: Dict[str, Type[ChessBot]] = {}
        self.bot_info_cache: Dict[str, Dict[str, Any]] = {}

        # Discover all bots from files
        self._discover_bots()

    def _discover_bots(self) -> None:
        """Automatically discover all bot files in the bots directory."""
        if not self.bots_directory.exists():
            logger.warning("Bots directory '%s' not found", self.bots_directory)
            return

        # Every public module of the directory may contain bots. Modules whose name starts with
        # an underscore (e.g. ``_search_bots.py``) hold shared base classes and are skipped.
        excluded = {"__init__.py", "chess_bot.py", "bot_registry.py"}
        bot_files = sorted(
            path
            for path in self.bots_directory.glob("*.py")
            if path.name not in excluded and not path.name.startswith("_")
        )

        for bot_file in bot_files:
            try:
                self._scan_file_for_bots(bot_file)
            except Exception as e:
                logger.warning("Error scanning %s: %s", bot_file, e)

    def _scan_file_for_bots(self, file_path: Path) -> None:
        """
        Scan a Python file for ChessBot implementations.

        :param file_path: Path to Python file to scan
        :type file_path: Path
        """
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(
                f"scanned_bot_{file_path.stem}", file_path
            )
            if spec is None or spec.loader is None:
                return

            module = importlib.util.module_from_spec(spec)

            # Add the source root (the folder holding ``chess_engine``) to Python path temporarily
            old_path = sys.path[:]
            sys.path.insert(0, str(self.bots_directory.parent.parent))

            try:
                spec.loader.exec_module(module)
            finally:
                sys.path[:] = old_path

            # Look for ChessBot subclasses in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    obj != ChessBot
                    and issubclass(obj, ChessBot)
                    and obj.__module__ == module.__name__
                ):
                    bot_name = name

                    # Avoid conflicts with other bots of the same name
                    if bot_name in self.bot_classes:
                        bot_name = f"{name}_{file_path.stem}"

                    self.bot_classes[bot_name] = obj
                    self.available_bots[bot_name] = str(file_path)

                    logger.debug("Found bot: %s in %s", bot_name, file_path.name)

        except Exception as e:
            logger.warning("Could not load bots from %s: %s", file_path.name, e)

    def list_available_bots(self) -> List[str]:
        """
        Get list of all available bot names.

        :return: List of bot names that can be used in games
        :rtype: List[str]
        """
        return sorted(self.bot_classes.keys())

    def list_loaded_bots(self) -> List[str]:
        """
        Get list of currently loaded (instantiated) bots.

        :return: List of bot names that are currently loaded in memory
        :rtype: List[str]
        """
        return list(self.loaded_bots.keys())

    def is_bot_available(self, bot_name: str) -> bool:
        """
        Check if a bot is available to be loaded.

        :param bot_name: Name of the bot to check
        :type bot_name: str
        :return: True if bot exists and can be loaded, False otherwise
        :rtype: bool
        """
        return bot_name in self.bot_classes

    def get_bot(self, bot_name: str, **options: Any) -> Optional[ChessBot]:
        """
        Get a bot instance, loading it if necessary.

        Without ``options`` the instance is cached and shared. With ``options`` (constructor
        keyword arguments such as ``model="cnn_resnet"`` or ``max_depth=4``) a fresh, uncached
        instance is built.

        :param bot_name: Name of the bot to get
        :type bot_name: str
        :param options: Keyword arguments forwarded to the bot constructor
        :return: ChessBot instance, or None if bot not found/failed to load
        :rtype: Optional[ChessBot]
        """
        if bot_name not in self.bot_classes:
            return None

        if options:
            return self._load_bot(bot_name, cache=False, **options)

        # Return already loaded bot
        if bot_name in self.loaded_bots:
            return self.loaded_bots[bot_name]

        return self._load_bot(bot_name)

    def _load_bot(self, bot_name: str, cache: bool = True, **options: Any) -> Optional[ChessBot]:
        """
        Load (instantiate) a bot class.

        :param bot_name: Name of bot to load
        :type bot_name: str
        :param cache: Remember the instance for later calls to :meth:`get_bot`
        :type cache: bool
        :param options: Keyword arguments forwarded to the bot constructor
        :return: ChessBot instance, or None if loading failed
        :rtype: Optional[ChessBot]
        """
        try:
            bot_class = self.bot_classes[bot_name]
            bot_instance = bot_class(**options)

            # Validate that the bot is properly implemented
            if not isinstance(bot_instance, ChessBot):
                logger.error("%s is not a proper ChessBot", bot_name)
                return None

            if not hasattr(bot_instance, "name") or not bot_instance.name:
                logger.error("%s doesn't have a valid name", bot_name)
                return None

            if cache:
                self.loaded_bots[bot_name] = bot_instance
                self.bot_info_cache[bot_name] = bot_instance.get_info()

            logger.info("Loaded bot: %s by %s", bot_instance.name, bot_instance.author)
            return bot_instance

        except Exception as e:
            logger.warning("Could not load bot %s: %s", bot_name, e)
            return None

    def get_bot_info(self, bot_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a bot without loading it.

        :param bot_name: Name of bot to get info for
        :type bot_name: str
        :return: Dictionary with bot info, or None if not available
        :rtype: Optional[Dict[str, Any]]
        """
        # Return cached info if available
        if bot_name in self.bot_info_cache:
            return self.bot_info_cache[bot_name]

        # Try to get info from loaded bot
        if bot_name in self.loaded_bots:
            info = self.loaded_bots[bot_name].get_info()
            self.bot_info_cache[bot_name] = info
            return info

        # As last resort, try to instantiate bot to get info
        if bot_name in self.bot_classes:
            temp_bot = self._load_bot(bot_name, cache=False)
            if temp_bot is None:
                return None
            info = temp_bot.get_info()
            self.bot_info_cache[bot_name] = info
            return info

        return None

    def reload_bots(self) -> int:
        """
        Reload all bots from files (useful during development).

        :return: Number of bots successfully reloaded
        :rtype: int
        """
        # Clear loaded bots and caches
        self.loaded_bots.clear()
        self.available_bots.clear()
        self.bot_classes.clear()
        self.bot_info_cache.clear()

        # Re-discover all bots
        self._discover_bots()
        new_count = len(self.bot_classes)

        logger.info("Reloaded bots: found %d total bots", new_count)
        return new_count

    def validate_all_bots(self) -> Dict[str, bool]:
        """
        Validate all available bots.

        :return: Dictionary mapping bot names to validation results
        :rtype: Dict[str, bool]
        """
        results = {}

        for bot_name in self.list_available_bots():
            try:
                bot = self.get_bot(bot_name)
                if bot is None:
                    results[bot_name] = False
                    continue

                # Basic validation checks
                valid = (
                    hasattr(bot, "name")
                    and hasattr(bot, "author")
                    and hasattr(bot, "get_move")
                    and callable(bot.get_move)
                )

                results[bot_name] = valid

            except Exception as e:
                logger.warning("Validation error for %s: %s", bot_name, e)
                results[bot_name] = False

        return results

    def get_tournament_ready_bots(self) -> List[str]:
        """
        Get list of bots that are ready for tournament play.

        :return: List of bot names that passed validation
        :rtype: List[str]
        """
        validation_results = self.validate_all_bots()
        return [
            bot_name
            for bot_name, is_valid in validation_results.items()
            if is_valid and bot_name != "HumanPlayer"
        ]  # Exclude human from tournaments

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the bot registry.

        :return: Dictionary with registry statistics
        :rtype: Dict[str, Any]
        """
        return {
            "total_available": len(self.bot_classes),
            "currently_loaded": len(self.loaded_bots),
            "builtin_bots": 1,  # HumanPlayer
            "custom_bots": len(self.bot_classes) - 1,
            "bot_files_found": len(self.available_bots),
            "tournament_ready": len(self.get_tournament_ready_bots()),
        }

    def __str__(self) -> str:
        """
        String representation of the registry.

        :return: String representation
        :rtype: str
        """
        stats = self.get_stats()
        return (
            f"BotRegistry: {stats['total_available']} bots available, "
            f"{stats['currently_loaded']} loaded, "
            f"{stats['tournament_ready']} tournament-ready"
        )

    def __repr__(self) -> str:
        """
        Developer representation of the registry.

        :return: Developer representation
        :rtype: str
        """
        return f"BotRegistry(bots_directory='{self.bots_directory}')"
