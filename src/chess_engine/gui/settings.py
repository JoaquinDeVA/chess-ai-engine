"""What the setup screen chooses: who plays each colour and how each AI is tuned. No Qt in here."""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

#: Techniques of selective search the AI players use by default: the ones that won in the games
#: against the reference search (docs/algorithms/selective-techniques.md). ``lmr`` is left out.
DEFAULT_TECHNIQUES = ("incremental", "tt_move", "history", "pvs", "aspiration", "null_move")

#: Threads an AI gets by default against a person, at most.
DEFAULT_THREADS = max(1, min(4, os.cpu_count() or 1))
#: Deepest iteration the window lets a person set; the default is no limit (the clock decides).
MAX_SETTABLE_DEPTH = 30

#: Name shown on the board for an AI.
AI_NAME = "AI"


@dataclass(frozen=True)
class AISettings:
    """How an AI player thinks (the options of the settings dialog).

    :param max_depth: Deepest iteration of the search; ``None`` lets the time decide.
    :param move_time: Seconds per move, or None to spend a share of the clock.
    :param opening_book: Play the book while the game follows it.
    :param tablebase: Play endings of up to five pieces from the tables.
    :param techniques: Selective-search techniques to switch on.
    :param threads: Threads that search.
    """

    max_depth: Optional[int] = None
    move_time: Optional[float] = 3.0
    opening_book: bool = True
    tablebase: bool = True
    techniques: Tuple[str, ...] = DEFAULT_TECHNIQUES
    threads: int = DEFAULT_THREADS


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
        return "You" if self.ai is None else AI_NAME


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
