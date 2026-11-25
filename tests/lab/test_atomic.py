"""Tests of the atomic write: a cut leaves the previous file, never a half one."""

import pytest

from chess_lab import atomic
from chess_lab.atomic import atomic_path
from chess_lab.measure import roundrobin
from tests.conftest import random_net


def test_the_file_appears_whole_under_its_name(tmp_path):
    path = tmp_path / "sub" / "out.txt"
    with atomic_path(path) as temporary:
        temporary.write_text("done")
        assert not path.exists()
    assert path.read_text() == "done"
    assert not list(path.parent.glob("*.tmp"))


def test_an_error_while_writing_keeps_the_previous_file_and_leaves_no_temporary(tmp_path):
    path = tmp_path / "out.txt"
    path.write_text("old")
    with pytest.raises(RuntimeError), atomic_path(path) as temporary:
        temporary.write_text("half")
        raise RuntimeError("cut")
    assert path.read_text() == "old"
    assert not list(tmp_path.glob("*.tmp"))


def test_a_cut_while_saving_a_model_keeps_the_previous_weights(tmp_path, monkeypatch):
    model = random_net(seed=1)
    path = model.save(tmp_path / "net.npz")
    before = path.read_bytes()

    def cut(*_):
        raise OSError("cut")

    monkeypatch.setattr(atomic.os, "replace", cut)
    with pytest.raises(OSError):
        random_net(seed=2).save(path)
    assert path.read_bytes() == before


def test_the_round_robin_state_is_written_atomically(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    roundrobin.write_atomic(path, '{"status": "running"}')

    def cut(*_):
        raise OSError("cut")

    monkeypatch.setattr(roundrobin.os, "replace", cut)
    with pytest.raises(OSError):
        roundrobin.write_atomic(path, "{")
    assert path.read_text() == '{"status": "running"}'
