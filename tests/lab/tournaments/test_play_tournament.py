"""Defaults of the tournament command line."""

from chess_lab.tournaments import play_tournament, standings


def test_play_tournament_defaults_use_h1_anchor():
    assert standings.ANCHOR == "h1"
    args = play_tournament.parse_args([])
    assert args.anchor == "h1"
    assert args.specs == ["linear_pst", "linear_pst_terms", "linear_king_bucket", "net4_pst", "h1"]
