"""Defaults shared by training, evaluation and the experiment scripts.

One place for the values that must agree everywhere: the seed of every sample and the share of mate
positions kept in training. The evaluation, the effort scale, the cache and the scripts import them from
here instead of keeping their own copies.
"""

#: Seed of every seeded sample (TRAIN, validation, test). The same seed gives the same positions.
SAMPLE_SEED = 42

#: Default share of the mate positions kept in training (the 100 % variant needs ``--tag``).
DEFAULT_MATE_KEEP = 0.10


def mates_suffix(mate_keep: float) -> str:
    """Name suffix of a representation that cuts the mates, e.g. ``"-mates10"``.

    :param mate_keep: Share of the mates kept, in ``(0, 1]``; 1 keeps them all.
    :return: ``""`` when every mate is kept.
    """
    return f"-mates{100 * mate_keep:g}" if mate_keep < 1.0 else ""
