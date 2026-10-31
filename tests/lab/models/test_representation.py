"""Target representations: score scale, point of view and the mate share kept."""

import chess
import numpy as np
import pandas as pd
import pytest

from chess_lab.models.representation import (
    LICHESS_K,
    Representation,
    is_quiet,
    keep_mate_rows,
    score_to_cp,
    target,
    white_score,
)

NAN = np.array([np.nan])


def test_winprob_zero_cp_is_half():
    assert white_score(np.array([0.0]), NAN, Representation())[0] == 0.5


def test_winprob_matches_lichess_curve():
    s = white_score(np.array([300.0]), NAN, Representation())[0]
    assert s == pytest.approx(1 / (1 + np.exp(-0.00368208 * 300)))


def test_cp_linear_clips_at_1267():
    rep = Representation(scale="cp_linear")
    s = white_score(np.array([5000.0, -1267.0]), np.full(2, np.nan), rep)
    assert s[0] == 1.0
    assert s[1] == 0.0


@pytest.mark.parametrize("scale", ["winprob", "cp_linear"])
def test_mates_are_one_or_zero_without_nan(scale):
    rep = Representation(scale=scale)
    s = white_score(np.array([np.nan, np.nan]), np.array([3.0, -1.0]), rep)
    assert s.tolist() == [1.0, 0.0]


def test_side_to_move_flips_for_black():
    cp, mate = np.array([200.0, 200.0]), np.full(2, np.nan)
    white_pov = white_score(cp, mate, Representation())
    t = target(cp, mate, np.array([True, False]), Representation())
    assert t[0] == white_pov[0]
    assert t[1] == pytest.approx(1 - white_pov[1])


def test_white_perspective_never_flips():
    rep = Representation(perspective="white")
    cp, mate = np.array([200.0, 200.0]), np.full(2, np.nan)
    t = target(cp, mate, np.array([True, False]), rep)
    assert t[0] == t[1] == white_score(cp, mate, rep)[0]


@pytest.mark.parametrize("scale", ["winprob", "cp_linear"])
def test_score_to_cp_inverts_white_score(scale):
    rep = Representation(scale=scale)
    cp = np.linspace(-800, 800, 33)
    back = score_to_cp(white_score(cp, np.full(cp.shape, np.nan), rep), rep)
    assert back == pytest.approx(cp, abs=1e-3)


def test_is_quiet():
    assert not is_quiet(chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1"), None)  # in check
    capture = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    assert not is_quiet(capture, "e4d5")
    ep = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert not is_quiet(ep, "e5d6")
    assert not is_quiet(chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"), "a7a8q")
    start = chess.Board()
    assert is_quiet(start, "e2e4")
    assert is_quiet(start, None)


def test_representation_round_trips_through_dict():
    rep = Representation(scale="cp_linear", perspective="white", filter="quiet")
    assert Representation.from_dict(rep.to_dict()) == rep
    assert Representation().name == "winprob-k0.00368-stm-all"
    assert LICHESS_K == 0.00368208


def test_mate_keep_is_nested():
    fens = pd.Series([f"fen-{i} w - - 0 1" for i in range(200_000)])
    keep_10 = keep_mate_rows(fens, 0.10)
    keep_5 = keep_mate_rows(fens, 0.05)
    assert (keep_5 <= keep_10).all()
    assert abs(keep_10.mean() - 0.10) < 0.01
    assert abs(keep_5.mean() - 0.05) < 0.01
    assert keep_mate_rows(fens, 1.0).all()


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_mate_keep_outside_zero_one_is_rejected(bad):
    with pytest.raises(ValueError):
        Representation(mate_keep=bad)
    with pytest.raises(ValueError):
        keep_mate_rows(pd.Series(["x"]), bad)


def test_mate_keep_is_part_of_the_name_only_when_not_default():
    assert Representation().name == "winprob-k0.00368-stm-all"
    assert Representation(mate_keep=1.0).name == "winprob-k0.00368-stm-all-mates100"
    assert Representation(mate_keep=0.05).name == "winprob-k0.00368-stm-all-mates5"
    rep = Representation(mate_keep=0.05)
    assert Representation.from_dict(rep.to_dict()) == rep
    assert Representation().to_dict()["mate_keep"] == 0.10
