"""Defaults shared by training, evaluation and the experiment scripts.

One place for the values that must agree everywhere: the seeds, the data directories and the duration
format. It sits outside the stages (models, datasets, training, measure), so all of them import it.
"""

from chess_engine.paths import ROOT

#: Seed of every seeded sample (TRAIN, validation, test). The same seed gives the same positions.
SAMPLE_SEED = 42
#: Salt of the split hash and of the mate subsampling. Changing it changes the benchmark: keep it fixed.
SPLIT_SEED = 42

#: Raw Lichess shards and their sidecars, absolute so that any working directory finds them.
RAW_DIR = ROOT / "data" / "LichessPositions" / "raw"
SIDECAR_DIR = ROOT / "data" / "LichessPositions" / "sidecar"
#: Cache of encoded training batches, absolute like the rest.
CACHE_ROOT = ROOT / "data" / "cache"


def format_duration(seconds: float) -> str:
    """Format a duration as ``H:MM:SS``.

    :param seconds: Duration.
    :return: The text.
    """
    return f"{int(seconds) // 3600}:{int(seconds) // 60 % 60:02d}:{int(seconds) % 60:02d}"
