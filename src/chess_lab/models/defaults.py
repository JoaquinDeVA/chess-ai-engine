"""Defaults shared by training, evaluation and the experiment scripts.

One place for the values that must agree everywhere: the seeds, the data directories and the share of
mate positions kept in training. The evaluation, the effort scale, the cache and the scripts import them
from here instead of keeping their own copies.
"""

from chess_engine.paths import ROOT

#: Seed of every seeded sample (TRAIN, validation, test). The same seed gives the same positions.
SAMPLE_SEED = 42
#: Salt of the split hash and of the mate subsampling. Changing it changes the benchmark: keep it fixed.
SPLIT_SEED = 42

#: Raw Lichess shards and their sidecars, absolute so that any working directory finds them.
RAW_DIR = ROOT / "data" / "lichess" / "raw"
SIDECAR_DIR = ROOT / "data" / "lichess" / "sidecar"

#: Default share of the mate positions kept in training (the 100 % variant needs ``--tag``).
DEFAULT_MATE_KEEP = 0.10


def mates_suffix(mate_keep: float) -> str:
    """Name suffix of a representation whose mate share is not the default, e.g. ``"-mates5"``.

    :param mate_keep: Share of the mates kept, in ``(0, 1]``; 1 keeps them all.
    :return: ``""`` when ``mate_keep`` is :data:`DEFAULT_MATE_KEEP`; otherwise
        ``"-mates{100 * mate_keep}"`` (``"-mates100"`` when every mate is kept).
    """
    return "" if mate_keep == DEFAULT_MATE_KEEP else f"-mates{100 * mate_keep:g}"


def format_duration(seconds: float) -> str:
    """Format a duration as ``H:MM:SS``.

    :param seconds: Duration.
    :return: The text.
    """
    return f"{int(seconds) // 3600}:{int(seconds) // 60 % 60:02d}:{int(seconds) % 60:02d}"
