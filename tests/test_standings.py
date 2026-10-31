"""Tests for the scores of a tournament: Elo, summaries, report section and model cards."""

import math

import pytest

from chess_lab.comparison import standings


def game(white, black, winner, **extra):
    return {"white": white, "black": black, "winner": winner, **extra}


def test_summarize_scores_and_elo():
    results = [
        game("mlp_features", "h1", "black"),  # h1 wins
        game("h1", "mlp_features", "white"),  # h1 wins
        game("mlp_features", "h1", None),  # draw
        game("h1", "mlp_features", "black"),  # mlp_features wins
        game("m", "h1", "white"),  # m wins
        game("h1", "m", "white"),  # h1 wins
        game("m", "h1", "white"),  # m wins
        game("h1", "m", "black"),  # m wins
        game("m", "mlp_features", None, error="KeyError: boom"),  # excluded from every score
    ]
    summary = standings.summarize(results, anchor="h1")
    mlp = summary["players"]["mlp_features"]
    assert mlp["games"] == 4
    assert (mlp["wins"], mlp["draws"], mlp["losses"]) == (1, 1, 2)
    assert mlp["score"] == pytest.approx(37.5)
    assert mlp["elo_vs_anchor"] == pytest.approx(-400 * math.log10(1 / 0.375 - 1))

    m = summary["players"]["m"]
    assert m["games"] == 4
    assert (m["wins"], m["draws"], m["losses"]) == (3, 0, 1)
    assert m["score"] == pytest.approx(75.0)
    assert m["elo_vs_anchor"] == pytest.approx(-400 * math.log10(1 / 0.75 - 1))

    assert summary["players"]["h1"]["elo_vs_anchor"] == pytest.approx(0.0)
    lo, hi = mlp["ci95"]
    assert lo < 37.5 < hi


def test_summarize_counts_terminations_and_forfeits():
    results = [
        game("m", "h1", "white", termination="checkmate"),
        game("h1", "m", "white", termination="checkmate"),
        game("m", "h1", "black", termination="time_forfeit"),
        game("h1", "m", "white", termination="player_error"),
        game("m", "h1", None, error="KeyError: boom"),  # crashed: not counted
    ]
    summary = standings.summarize(results, anchor="h1")
    assert summary["terminations"] == {"checkmate": 2, "time_forfeit": 1, "player_error": 1}
    assert summary["forfeit_games"] == 2
    text = "\n".join(standings.termination_section(summary))
    assert "## How the games ended" in text
    assert "| checkmate | 2 | 50.0% |" in text
    assert "Games decided by a forfeit (time, player error or illegal move): 2 of 4." in text
    assert "No games." in "\n".join(
        standings.termination_section({"terminations": {}, "forfeit_games": 0})
    )


def test_iterative_fit_matches_direct_score_for_two_players():
    results = [
        game("mlp_features", "h1", "black"),
        game("h1", "mlp_features", "white"),
        game("mlp_features", "h1", None),
        game("h1", "mlp_features", "black"),
    ]
    mlp = standings.summarize(results, anchor="h1")["players"]["mlp_features"]
    assert mlp["elo"] == pytest.approx(mlp["elo_vs_anchor"], abs=0.5)


def test_summarize_requires_the_anchor_player():
    with pytest.raises(ValueError):
        standings.summarize([{"white": "mlp_features", "black": "m", "winner": None}])


def test_size_for_budget():
    # 1 h, 2 workers, 10 s per game -> 720 games; 3 players -> 3 pairs -> 6 games per opening.
    assert standings.size_for_budget(hours=1, workers=2, n_players=3, seconds_per_game=10) == 120
    # 10 h, 5 workers, 10 s per game -> 18000 games; 5 players -> 10 pairs -> 20 per opening.
    assert standings.size_for_budget(hours=10, workers=5, n_players=5, seconds_per_game=10) == 900


def test_record_cards_writes_tournament_sections_shown_by_the_history(tmp_path, monkeypatch):
    from chess_engine.models import cards
    from chess_lab.comparison import report

    monkeypatch.setattr(cards, "CARDS_DIR", tmp_path / "cards")
    cards.update_card("m", status="current", architecture="A")
    results = [
        game("m", "h1", "white"),
        game("h1", "m", "white"),
        game("m", "mlp_features", None),
        game("mlp_features", "m", "black"),
        game("h1", "mlp_features", "white"),
        game("m2", "h1", "black"),
    ]
    summary = standings.summarize(results, anchor="h1")
    assert summary["players"]["m"]["elo_interval"] > 0

    recorded = standings.record_cards(summary, ["mlp_features", "h1", "m", "m2"], "h1", 10.0, 0.1)
    assert recorded == ["m"]
    section = cards.load_card("m")["tournament"]
    entry = summary["players"]["m"]
    assert section == {"10+0.1": _entry(entry, "h1", 10.0, 0.1)}
    assert not cards.card_path("m2").exists() and not cards.card_path("h1").exists()
    row = next(line for line in report.render_tables().splitlines() if "`m`" in line)
    assert f"{entry['score']:.1f}%" in row and "±" in row


def _entry(player, anchor, base, increment, max_depth=standings.DEFAULT_MAX_DEPTH):
    return {
        "games": player["games"],
        "score": player["score"],
        "elo": player["elo"],
        "elo_interval": player["elo_interval"],
        "anchor": anchor,
        "base": base,
        "increment": increment,
        "max_depth": max_depth,
    }


def _two_player_summary():
    return standings.summarize([game("m", "h1", "white"), game("h1", "m", "white")], anchor="h1")


def test_control_label_rule():
    assert standings.control_label(60, 1) == "60+1"
    assert standings.control_label(600.0, 5.0) == "600+5"
    assert standings.control_label(10, 0.1) == "10+0.1"
    assert standings.control_label(60, 1, max_depth=3) == "depth3"
    assert standings.control_label(60, 1, max_depth=standings.DEFAULT_MAX_DEPTH) == "60+1"


def test_record_cards_keeps_one_entry_per_control(tmp_path, monkeypatch):
    from chess_engine.models import cards

    monkeypatch.setattr(cards, "CARDS_DIR", tmp_path / "cards")
    cards.update_card("m", status="current")
    summary = _two_player_summary()
    standings.record_cards(summary, ["m", "h1"], "h1", 60, 1)
    standings.record_cards(summary, ["m", "h1"], "h1", 600, 5)
    standings.record_cards(summary, ["m", "h1"], "h1", 60, 1, max_depth=3)
    standings.record_cards(summary, ["m", "h1"], "h1", 60, 1)  # same control: replaced
    section = cards.load_card("m")["tournament"]
    assert set(section) == {"60+1", "600+5", "depth3"}
    assert section["depth3"]["max_depth"] == 3 and section["600+5"]["base"] == 600


def test_record_cards_migrates_an_old_flat_section(tmp_path, monkeypatch):
    from chess_engine.models import cards

    monkeypatch.setattr(cards, "CARDS_DIR", tmp_path / "cards")
    flat = {"games": 8, "score": 55.0, "elo": 10.0, "elo_interval": 5.0,
            "anchor": "h1", "base": 60.0, "increment": 1.0}  # fmt: skip
    cards.update_card("m", status="current", tournament=flat)
    standings.record_cards(_two_player_summary(), ["m"], "h1", 600, 5)
    section = cards.load_card("m")["tournament"]
    assert section["60+1"] == flat
    assert set(section) == {"60+1", "600+5"}  # no stale flat keys at the top level


def test_record_cards_records_h1_only_when_its_card_exists(tmp_path, monkeypatch):
    from chess_engine.models import cards

    monkeypatch.setattr(cards, "CARDS_DIR", tmp_path / "cards")
    cards.update_card("m", status="current")
    summary = _two_player_summary()
    assert standings.record_cards(summary, ["m", "h1"], "h1", 60, 1) == ["m"]
    cards.update_card("h1", status="reference")
    assert standings.record_cards(summary, ["m", "h1"], "h1", 60, 1) == ["m", "h1"]
    assert list(cards.load_card("h1")["tournament"]) == ["60+1"]
