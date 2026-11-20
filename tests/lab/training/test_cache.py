"""Directory of the encoded-batch cache."""

from chess_lab.models.definitions import NnThreatsW64BothModel, NnW64BothModel, NnW128BothModel
from chess_lab.models.target import Representation
from chess_lab.training.cache import cache_path


def test_cache_path_changes_with_every_input():
    base = {"model_class": NnW64BothModel, "rep": Representation(mate_keep=0.10)}
    args = {"positions": 1000, "seed": 42, "quiet_only": False, "batch_size": 4096}
    reference = cache_path(**base, **args)

    assert cache_path(**base, **args) == reference
    assert cache_path(**base, **{**args, "positions": 2000}) != reference
    assert cache_path(**base, **{**args, "seed": 7}) != reference
    assert cache_path(**base, **{**args, "quiet_only": True}) != reference
    assert cache_path(**base, **{**args, "batch_size": 2048}) != reference
    assert (
        cache_path(model_class=NnW64BothModel, rep=Representation(mate_keep=0.5), **args)
        != reference
    )
    assert cache_path(model_class=NnW64BothModel, rep=Representation(k=0.01), **args) != reference


def test_cache_path_depends_on_feature_set_not_on_class():
    args = {"rep": Representation(), "positions": 1000, "seed": 42, "quiet_only": True}
    args["batch_size"] = 4096

    assert cache_path(model_class=NnW128BothModel, **args) == cache_path(
        model_class=NnW64BothModel, **args
    )
    assert cache_path(model_class=NnThreatsW64BothModel, **args) != cache_path(
        model_class=NnW64BothModel, **args
    )
