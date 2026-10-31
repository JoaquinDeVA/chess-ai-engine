"""Scores of a round-robin tournament: Elo, W-D-L and intervals, report section and model cards.

Everything here works on the game records written by :mod:`chess_lab.comparison.tournament`;
nothing plays a game.
"""

import logging
import math
import statistics
from typing import Any, Dict, List, Sequence, Tuple

from chess_engine.game.play import FORFEIT_TERMINATIONS
from chess_lab.comparison.tradeoff import HEURISTICS
from chess_lab.models.cards import card_path, load_card, save_card

logger = logging.getLogger(__name__)

#: Player whose rating is fixed at 0 Elo.
ANCHOR = "h1"

#: Scores are clipped to ``[CLIP, 1 - CLIP]`` before taking a logarithm.
CLIP = 0.01

#: Deepest iteration of a player when the tournament does not limit it (see make_player).
DEFAULT_MAX_DEPTH = 30


def _elo_from_score(score: float) -> float:
    """Elo difference implied by a score, ``-400 * log10(1/s - 1)`` with ``s`` clipped."""
    s = min(max(score, CLIP), 1 - CLIP)
    return -400 * math.log10(1 / s - 1)


def _fit_elo(
    counts: Dict[Tuple[str, str], int],
    points: Dict[str, float],
    games: Dict[str, int],
    anchor: str,
) -> Dict[str, float]:
    """Bradley-Terry ratings from all pairs, fitted by the minorise-maximise iteration.

    Scores per player are clipped like the pairwise ones, so no rating diverges. The ratings are
    anchored at ``anchor``.

    :param counts: Number of games per ordered pair ``(player, opponent)``.
    :param points: Points (wins + half draws) per player.
    :param games: Games per player.
    :param anchor: Player whose rating is fixed at 0.
    :return: Elo per player with games; empty if the anchor has no games.
    """
    players = [p for p in games if games[p] > 0]
    if anchor not in players:
        return {}
    weight = {p: min(max(points[p] / games[p], CLIP), 1 - CLIP) * games[p] for p in players}
    gamma = {p: 1.0 for p in players}
    for _ in range(2000):
        updated = {}
        for p in players:
            denominator = sum(
                counts[(p, q)] / (gamma[p] + gamma[q])
                for q in players
                if q != p and counts.get((p, q))
            )
            updated[p] = weight[p] / denominator
        scale = updated[anchor]
        change = max(abs(updated[p] / scale - gamma[p]) for p in players)
        gamma = {p: updated[p] / scale for p in players}
        if change < 1e-10:
            break
    return {p: 400 * math.log10(gamma[p]) for p in players}


def summarize(results: Sequence[Dict[str, Any]], anchor: str = ANCHOR) -> Dict[str, Any]:
    """Scores, W-D-L, Elo and 95% intervals of a set of game records.

    Crashed games (records with ``"error"``) are left out. For each player the score is
    ``wins + draws / 2`` over games. ``elo_vs_anchor`` uses the player's score against ``anchor``
    alone; ``elo`` is the Bradley-Terry fit over all pairs. The 95% interval of the score is the
    normal approximation from the variance of the per-game points; ``elo_interval`` is the
    half-width of the Elo implied by that score interval (an approximation, not a fitted interval).

    Caveats: scores entering the Elo formulas are clipped to ``[0.01, 0.99]``. The intervals treat
    games as independent, but the two games of a paired opening are correlated, so they are
    slightly optimistic. Crashed games are not retried.

    :param results: Game records as written by :func:`run_round_robin`.
    :param anchor: Player whose Elo is 0.
    :return: ``players`` (per player: games, wins, draws, losses, score in percent, ``ci95`` in
        percent, ``elo_vs_anchor``, ``elo``, ``elo_interval``), ``head_to_head`` (``[wins, draws,
        losses]`` of the row player against the column player), ``errors`` (number of crashed
        games), ``terminations`` (games per way of ending, crashed games excluded) and
        ``forfeit_games`` (games ended by a forfeit; they still count as normal results).
    :raises ValueError: If ``anchor`` takes part in none of the records.
    """
    if not any(anchor in (r["white"], r["black"]) for r in results):
        raise ValueError(f"the anchor player '{anchor}' must take part in the tournament")
    players = sorted({p for r in results for p in (r["white"], r["black"])})
    per_game: Dict[str, List[float]] = {p: [] for p in players}
    head = {p: {q: [0, 0, 0] for q in players if q != p} for p in players}
    counts: Dict[Tuple[str, str], int] = {}
    errors = 0
    terminations: Dict[str, int] = {}

    for record in results:
        if "error" in record:
            errors += 1
            continue
        if "termination" in record:
            terminations[record["termination"]] = terminations.get(record["termination"], 0) + 1
        white, black = record["white"], record["black"]
        white_points = {"white": 1.0, "black": 0.0, None: 0.5}[record["winner"]]
        per_game[white].append(white_points)
        per_game[black].append(1.0 - white_points)
        counts[(white, black)] = counts.get((white, black), 0) + 1
        counts[(black, white)] = counts.get((black, white), 0) + 1
        for player, opponent, points in (
            (white, black, white_points),
            (black, white, 1.0 - white_points),
        ):
            slot = 0 if points == 1.0 else 1 if points == 0.5 else 2
            head[player][opponent][slot] += 1

    games = {p: len(per_game[p]) for p in players}
    points = {p: sum(per_game[p]) for p in players}
    fitted = _fit_elo(counts, points, games, anchor)

    stats: Dict[str, Dict[str, Any]] = {}
    for p in players:
        n = games[p]
        entry: Dict[str, Any] = {
            "games": n,
            "wins": sum(record[0] for record in head[p].values()),
            "draws": sum(record[1] for record in head[p].values()),
            "losses": sum(record[2] for record in head[p].values()),
            "score": None,
            "ci95": None,
            "elo_vs_anchor": None,
            "elo": fitted.get(p),
            "elo_interval": None,
        }
        if n:
            mean = points[p] / n
            variance = statistics.fmean(x * x for x in per_game[p]) - mean * mean
            margin = 1.96 * math.sqrt(max(variance, 0.0) / n)
            entry["score"] = 100 * mean
            low, high = max(0.0, mean - margin), min(1.0, mean + margin)
            entry["ci95"] = [100 * low, 100 * high]
            entry["elo_interval"] = (_elo_from_score(high) - _elo_from_score(low)) / 2
            if p == anchor:
                entry["elo_vs_anchor"] = 0.0
            elif counts.get((p, anchor)):
                wins, draws, losses = head[p][anchor]
                score = (wins + 0.5 * draws) / (wins + draws + losses)
                entry["elo_vs_anchor"] = _elo_from_score(score)
        stats[p] = entry

    return {
        "players": stats,
        "head_to_head": head,
        "errors": errors,
        "terminations": terminations,
        "forfeit_games": sum(n for t, n in terminations.items() if t in FORFEIT_TERMINATIONS),
    }


def termination_section(summary: Dict[str, Any]) -> List[str]:
    """Markdown lines of the "How the games ended" section of the report.

    :param summary: Output of :func:`summarize`.
    :return: Lines of the section, to be placed right after the cross table.
    """
    terminations = summary["terminations"]
    total = sum(terminations.values())
    lines = ["## How the games ended", ""]
    if total:
        lines += ["| Termination | Games | Share |", "|---|---|---|"]
        for kind, n in sorted(terminations.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| {kind} | {n} | {100 * n / total:.1f}% |")
    else:
        lines.append("No games.")
    lines += [
        "",
        "Games decided by a forfeit (time, player error or illegal move): "
        f"{summary['forfeit_games']} of {total}. These are counted as results in the scores above.",
        "",
    ]
    return lines


def control_label(base: float, increment: float, max_depth: int = DEFAULT_MAX_DEPTH) -> str:
    """Key of a tournament in the ``tournament`` section of a card.

    :param base: Starting clock, in seconds.
    :param increment: Seconds per move.
    :param max_depth: Depth limit; a limited match is labelled by its depth, not by its clock.
    :return: ``depthN`` for a depth-limited match, else ``base+increment`` (e.g. ``60+1``).
    """
    if max_depth != DEFAULT_MAX_DEPTH:
        return f"depth{max_depth}"
    return f"{base:g}+{increment:g}"


def record_cards(
    summary: Dict[str, Any],
    specs: Sequence[str],
    anchor: str,
    base: float,
    increment: float,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> List[str]:
    """Write the entry of this tournament in the ``tournament`` section of the cards.

    The section holds one entry per control (see :func:`control_label`); the other controls
    are kept. An old flat section (one tournament, with ``base``) is migrated first. A player
    whose card does not exist yet is skipped (cards are written by training and by
    ``comparison.evaluate``, not created here), and so is one that played no game.

    :param summary: Output of :func:`summarize`.
    :param specs: Players of the tournament.
    :param anchor: Elo anchor of the summary.
    :param base: Starting clock, in seconds.
    :param increment: Seconds per move.
    :param max_depth: Depth limit of every player.
    :return: The players whose card was updated.
    """
    label = control_label(base, increment, max_depth)
    recorded = []
    for spec in specs:
        entry = summary["players"].get(spec)
        if not entry or not entry["games"]:
            continue
        if not card_path(spec).exists():
            if spec not in HEURISTICS:
                logger.warning("no model card for %s: tournament section not written", spec)
            continue
        card = load_card(spec)
        section = card.get("tournament") or {}
        if "base" in section:  # old layout: a single flat tournament
            old = section
            section = {control_label(old["base"], old.get("increment", 0.0)): old}
        section[label] = {
            "games": entry["games"],
            "score": entry["score"],
            "elo": entry["elo"],
            "elo_interval": entry["elo_interval"],
            "anchor": anchor,
            "base": base,
            "increment": increment,
            "max_depth": max_depth,
        }
        card["tournament"] = section
        save_card(spec, card)
        recorded.append(spec)
    return recorded


def size_for_budget(hours: float, workers: int, n_players: int, seconds_per_game: float) -> int:
    """Openings per pair that fit a time budget.

    One opening costs ``2 * pairs`` games, where ``pairs`` is the number of unordered pairs.

    :param hours: Wall-clock budget.
    :param workers: Games played at the same time.
    :param n_players: Number of players.
    :param seconds_per_game: Worker-seconds per game (wall time times workers over games).
    :return: Number of openings (rounded down; may be 0).
    :raises ValueError: If there are fewer than two players.
    """
    pairs = n_players * (n_players - 1) // 2
    if pairs == 0:
        raise ValueError("a round-robin needs at least two players")
    games_available = hours * 3600.0 * workers / seconds_per_game
    return int(games_available // (2 * pairs))
