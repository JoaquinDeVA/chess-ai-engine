"""What the setup screen chooses: who plays each colour and how each AI is tuned. No Qt in here."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from chess_engine import _core
from chess_engine.bots._search_bots import DEFAULT_TECHNIQUES
from chess_engine.bots.bot_registry import BotRegistry
from chess_engine.bots.chess_bot import ChessBot
from chess_engine.bots.iterative_model_book_bot import DEFAULT_MODEL
from chess_engine.models.registry import artifact_path, available_models
from chess_engine.search.time_management import TimeManager

HEURISTIC = "h1"
HEURISTIC_BOT = "IterativeHeuristicBot"
MODEL_BOT = "IterativeModelBookBot"

#: Name shown for each evaluation next to its short id (docs/MODELS.md).
EVALUATION_NAMES: Dict[str, str] = {
    HEURISTIC: "Hand-written heuristic",
    "linear_material": "Material logistic model",
    "linear_pst": "Piece-square logistic model",
    "linear_pst_terms": "Piece-square-and-terms logistic model",
    "linear_king_bucket": "King-bucket logistic model",
    "net4_pst": "Four-layer piece-square network",
}

#: Threads a bot gets by default against a person, at most.
DEFAULT_THREADS = max(1, min(4, os.cpu_count() or 1))
#: Deepest iteration the window lets a person set; the default is no limit (the clock decides).
MAX_SETTABLE_DEPTH = 30


def evaluation_label(evaluation: str) -> str:
    """Name and id of an evaluation, e.g. ``Material logistic model (linear_material)``.

    :param evaluation: Short id.
    :return: Text for a menu.
    """
    name = EVALUATION_NAMES.get(evaluation)
    return f"{name} ({evaluation})" if name else evaluation


def available_evaluations() -> List[str]:
    """Evaluations that can be played now: the heuristic and the models whose file exists.

    :return: Short ids, the default model first and the heuristic last.
    """
    models = [name for name in available_models() if artifact_path(name).exists()]
    models.sort(key=lambda name: (name != DEFAULT_MODEL, name))
    return models + [HEURISTIC]


@dataclass(frozen=True)
class AISettings:
    """How an AI player thinks (the options of the settings dialog).

    :param evaluation: ``"h1"`` or the id of a model.
    :param max_depth: Deepest iteration of the search; ``None`` lets the time decide.
    :param move_time: Seconds per move, or None to spend a share of the clock.
    :param opening_book: Play the book while the game follows it.
    :param tablebase: Play endings of up to five pieces from the tables.
    :param quiescence_plies: Plies of capture search at the horizon (0 = off).
    :param techniques: Selective-search techniques to switch on.
    :param threads: Threads that search.
    """

    evaluation: str = DEFAULT_MODEL
    max_depth: Optional[int] = None
    move_time: Optional[float] = 3.0
    opening_book: bool = True
    tablebase: bool = True
    quiescence_plies: int = _core.DEFAULT_QUIESCENCE_PLIES
    techniques: Tuple[str, ...] = tuple(DEFAULT_TECHNIQUES)
    threads: int = DEFAULT_THREADS

    @property
    def bot_name(self) -> str:
        """Registered bot that plays this evaluation."""
        return HEURISTIC_BOT if self.evaluation == HEURISTIC else MODEL_BOT

    def bot_options(self) -> Dict[str, Any]:
        """Constructor options for :meth:`BotRegistry.get_bot`.

        :return: Keyword arguments of the bot.
        """
        options: Dict[str, Any] = {
            "max_depth": self.max_depth,
            "quiescence_plies": self.quiescence_plies,
            "techniques": tuple(self.techniques),
            "threads": self.threads,
        }
        if self.evaluation != HEURISTIC:
            options["model"] = self.evaluation
        if not self.opening_book:
            options["opening_book"] = None
        if not self.tablebase:
            options["tablebase"] = None
        if self.move_time is not None:
            options["time_manager"] = TimeManager(
                min_turn_time=self.move_time, max_turn_time=self.move_time
            )
        return options


@dataclass(frozen=True)
class PlayerSetup:
    """Who plays a colour: a person (``ai`` is None) or an AI with its settings."""

    ai: Optional[AISettings] = None

    @property
    def is_human(self) -> bool:
        """True for a person at the board."""
        return self.ai is None

    @property
    def label(self) -> str:
        """Name shown on the board."""
        return "You" if self.ai is None else EVALUATION_NAMES.get(self.ai.evaluation, "AI")


@dataclass(frozen=True)
class MatchSetup:
    """A game to play.

    :param white: Player of White.
    :param black: Player of Black.
    :param base: Seconds each side starts with, or None for no clock.
    :param increment: Seconds added after each move.
    :param fen: Starting position (default: the initial one).
    """

    white: PlayerSetup = field(default_factory=PlayerSetup)
    black: PlayerSetup = field(default_factory=lambda: PlayerSetup(AISettings()))
    base: Optional[float] = 600.0
    increment: float = 5.0
    fen: Optional[str] = None

    def has_humans(self) -> bool:
        """True if somebody sits at the board."""
        return self.white.is_human or self.black.is_human


def build_bot(settings: AISettings, registry: Optional[BotRegistry] = None) -> ChessBot:
    """Create the bot for some settings.

    :param settings: Choices of the settings dialog.
    :param registry: Registry to take the bot class from (default: a new one).
    :return: A new bot, not shared with anybody.
    :raises RuntimeError: If the registry cannot build it (e.g. the model file is missing).
    """
    registry = registry or BotRegistry()
    bot = registry.get_bot(settings.bot_name, **settings.bot_options())
    if bot is None:
        raise RuntimeError(f"could not build the bot for {settings.evaluation}")
    return bot
