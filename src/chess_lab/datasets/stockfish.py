"""Stockfish through ``chess.engine``: open it configured, read a score, label positions in parallel."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Callable, Iterator, Optional, TypeVar

import chess.engine

T = TypeVar("T")

#: Chunks of work per process, so a slow chunk does not hold the others back.
CHUNKS_PER_JOB = 8


@contextmanager
def open_engine(
    path: str, threads: int = 1, hash_mb: Optional[int] = 64
) -> Iterator[chess.engine.SimpleEngine]:
    """Open a UCI engine, configure it and quit it on exit.

    :param path: Path of the engine binary.
    :param threads: Value of the ``Threads`` option.
    :param hash_mb: Value of the ``Hash`` option in MB; None leaves the engine's default.
    :return: The open engine.
    """
    engine = chess.engine.SimpleEngine.popen_uci(path)
    try:
        options: dict = {"Threads": threads}
        if hash_mb is not None:
            options["Hash"] = hash_mb
        engine.configure(options)
        yield engine
    finally:
        engine.quit()


def cp_mate(score: chess.engine.Score) -> tuple[float, float]:
    """A score as ``(cp, mate)``: the one that does not apply is NaN.

    :param score: Score from the point of view the caller chose (``.white()`` or ``.pov(color)``).
    :return: ``(cp, nan)``, or ``(nan, moves to mate)`` with the sign of the side that mates.
    """
    mate = score.mate()
    if mate is not None:
        return float("nan"), float(mate)
    return float(score.score()), float("nan")


def _label_chunk(args: tuple[str, int, list[str], Callable]) -> list[tuple[str, object]]:
    """Worker: label a list of positions with its own engine."""
    path, depth, fens, label = args
    with open_engine(path) as engine:
        return [(fen, label(engine, fen, depth)) for fen in fens]


def label_parallel(
    path: str,
    depth: int,
    fens: list[str],
    jobs: int,
    label: Callable[[chess.engine.SimpleEngine, str, int], T],
) -> list[tuple[str, T]]:
    """Label positions with one engine process per job.

    :param path: Path of the engine binary.
    :param depth: Fixed search depth.
    :param fens: Positions to label.
    :param jobs: Engine processes.
    :param label: Module-level function ``(engine, fen, depth) -> result`` (it must be picklable).
    :return: ``(fen, result)`` per position, grouped by chunk: sort them if the order matters.
    """
    count = jobs * CHUNKS_PER_JOB
    chunks = [(path, depth, fens[i::count], label) for i in range(count)]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return [pair for part in pool.map(_label_chunk, chunks) for pair in part]
