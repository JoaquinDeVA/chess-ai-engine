"""Tests of the sorted hash index of position keys."""

import numpy as np

from chess_lab.datasets.fen_index import hash_keys, lookup, sorted_index


def test_lookup_gives_the_value_of_each_indexed_key_and_missing_for_the_rest():
    shards = [["a", "b"], ["c"]]
    hashes, values = sorted_index(
        [hash_keys(keys) for keys in shards], [np.array([1, 2], np.int8), np.array([3], np.int8)]
    )
    assert hashes.tolist() == sorted(hashes.tolist())
    assert lookup(hashes, values, ["c", "zz", "a", "b"], -1).tolist() == [3, -1, 1, 2]


def test_an_empty_index_finds_nothing():
    empty = np.array([], np.int64)
    assert lookup(empty, empty, ["a"], -1).tolist() == [-1]
