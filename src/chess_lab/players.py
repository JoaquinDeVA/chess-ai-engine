"""Players for the measurements and the tournaments: built from a spec, with no book or tables.

A spec is ``h1`` (the hand-written heuristic), a model name or ``name@tag``, optionally followed by
``+technique,...`` to switch techniques of selective search on.
"""

import os
from contextlib import contextmanager
from typing import Any, Iterator, Sequence, Tuple

from chess_engine import _core
from chess_engine.bots.search_bot import SearchBot
from chess_engine.bots.time_manager import TimeManager
from chess_engine.models import split_spec
from chess_lab.models.definitions import get_model_class

HEURISTICS = ("h1",)

#: Thread-pool variables pinned to 1 in the workers, so that timed games do not compete for cores.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)


@contextmanager
def single_threaded_env() -> Iterator[None]:
    """Set :data:`THREAD_VARIABLES` to 1 and restore the previous environment on exit.

    Workers spawned inside the block inherit the variables.
    """
    saved = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    os.environ.update({name: "1" for name in THREAD_VARIABLES})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def fixed_time_manager(seconds: float) -> TimeManager:
    """Time manager whose per-move budget is exactly ``seconds`` (no extension, no margin).

    :param seconds: Seconds allowed for each move.
    :return: The time manager.
    """
    return TimeManager(
        move_fraction=1.0,
        min_turn_time=seconds,
        max_turn_time=seconds,
        max_clock_share=1.0,
        extension=1.0,
        safety_margin=0.0,
    )


def split_techniques(spec: str) -> Tuple[str, Tuple[str, ...]]:
    """Split ``name+tech1,tech2`` into the player and the techniques of selective search.

    ``name+all`` switches every technique on. A spec without ``+`` has no techniques.

    :param spec: Player specification, possibly with a ``+`` suffix.
    :return: ``(player specification, techniques)``.
    """
    base, _, rest = spec.partition("+")
    if not rest:
        return base, ()
    names = list(_core.TECHNIQUES) if rest == "all" else [n for n in rest.split(",") if n]
    return base, tuple(names)


def make_player(
    spec: str,
    time_manager: TimeManager,
    max_depth: int = 30,
    **search_options: Any,
) -> SearchBot:
    """Build an iterative-deepening player for measurements: no opening book and no tablebases.

    :param spec: ``"h1"`` for a hand-crafted heuristic, a registered model
        name or ``name@tag`` (its player is named after the spec). A ``+`` suffix switches techniques of selective search on (``linear_pst+pvs,lmr``
        or ``linear_pst+all``); without it the player uses the reference search, so that measurements
        compare evaluations alone.
    :param time_manager: Time budgeting policy.
    :param max_depth: Deepest iteration to run.
    :param search_options: Other options of the bot (``estimator``, ``min_depth``...); they may
        switch the book (``opening_book``) or the tables (``tablebase``) on.
    :return: The player.
    """
    search_options = {"opening_book": None, "tablebase": None, **search_options}
    label = spec
    techniques: Tuple[str, ...] = ()  # the reference search, unless the spec asks for techniques
    if "+" in spec:
        spec, techniques = split_techniques(spec)
    search_options.setdefault("techniques", techniques)
    return SearchBot(
        spec, name=label, max_depth=max_depth, time_manager=time_manager, **search_options
    )


def validate_specs(specs: Sequence[str]) -> None:
    """Fail fast on a player that is neither a heuristic nor a registered model.

    :param specs: Player specifications (``name`` or ``name@tag``).
    :raises ValueError: If a spec is unknown.
    """
    for spec in specs:
        spec = split_techniques(spec)[0]
        if spec in HEURISTICS:
            continue
        try:
            get_model_class(split_spec(spec)[0])
        except (KeyError, ValueError) as error:
            raise ValueError(f"unknown player {spec!r}: {error.args[0]}") from error
