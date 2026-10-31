#!/usr/bin/env python3
"""Play two configurations of the search against each other and report who wins.

python src/chess_lab/scripts/bench/compare_engine_versions.py --model linear_pst_terms --candidate pvs,lmr --openings 50

The same evaluation model plays twice: the baseline with the techniques of ``--baseline`` (none by
default: the reference search) and the candidate with those of ``--candidate`` (``all`` for every
technique of selective search). Each opening is played twice with the colours swapped, at equal
time per move. The report gives the candidate's score with its 95 % interval, its record, its Elo
difference and the mean depth each side completed.

This compares versions of the engine, not models: it does not write the model cards and it is not
a tournament of models.
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Sequence

from chess_lab.tournaments.standings import DEFAULT_MAX_DEPTH, summarize
from chess_lab.tournaments.tournament import paired_openings, read_games, run_round_robin


def spec_of(model: str, techniques: str) -> str:
    """Player specification of a model with techniques (``model``, ``model+a,b``).

    :param model: Model or heuristic name.
    :param techniques: Comma-separated techniques, ``all`` or empty.
    :return: The specification understood by ``make_player``.
    """
    return f"{model}+{techniques}" if techniques else model


def mean_depths(log_dir: Path, specs: Sequence[str]) -> Dict[str, float]:
    """Mean completed depth per move of each player, from the per-move logs of the games.

    :param log_dir: Folder with one JSON log per game.
    :param specs: The two players.
    :return: ``{spec: mean completed depth}`` (0 for a player without moves).
    """
    depths: Dict[str, List[int]] = {spec: [] for spec in specs}
    for path in log_dir.glob("*.json"):
        game = json.loads(path.read_text())
        for move in game.get("moves", []):
            player = game[move["color"]]
            if player in depths and move.get("completed_depth"):
                depths[player].append(move["completed_depth"])
    return {spec: (statistics.fmean(d) if d else 0.0) for spec, d in depths.items()}


def render(
    baseline: str,
    candidate: str,
    summary: Dict,
    depths: Dict[str, float],
    base: float,
    increment: float,
) -> List[str]:
    """The report of a comparison.

    :param baseline: Specification of the baseline.
    :param candidate: Specification of the candidate.
    :param summary: Output of ``summarize`` anchored on the baseline.
    :param depths: Mean completed depths.
    :param base: Starting clock in seconds.
    :param increment: Seconds per move.
    :return: Lines of Markdown.
    """
    player = summary["players"][candidate]
    lines = [
        "# Engine versions compared",
        "",
        f"Baseline `{baseline}` against candidate `{candidate}` at {base:g} s + {increment:g} s.",
        "",
        "| | games | W-D-L | score % | 95 % interval % | Elo vs baseline | mean depth |",
        "|---|---|---|---|---|---|---|",
    ]
    low, high = player["ci95"]
    lines.append(
        f"| candidate `{candidate}` | {player['games']} | "
        f"{player['wins']}-{player['draws']}-{player['losses']} | {player['score']:.1f} | "
        f"{low:.1f} - {high:.1f} | {player['elo_vs_anchor']:+.0f} | {depths[candidate]:.2f} |"
    )
    lines.append(f"| baseline `{baseline}` | | | | | 0 | {depths[baseline]:.2f} |")
    return lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="linear_pst_terms", help="evaluation both sides use")
    parser.add_argument("--baseline", default="", help="techniques of the baseline (default none)")
    parser.add_argument("--candidate", default="all", help="techniques of the candidate")
    parser.add_argument("--openings", type=int, default=50, help="openings (two games each)")
    parser.add_argument("--base", type=float, default=10.0, help="starting clock, seconds")
    parser.add_argument("--increment", type=float, default=0.1, help="seconds per move")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--out", type=Path, help="games file (resumable)")
    parser.add_argument("--report", type=Path, help="Markdown report")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Play the games and write the report.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    baseline = spec_of(args.model, args.baseline)
    candidate = spec_of(args.model, args.candidate)
    if baseline == candidate:
        raise SystemExit("the baseline and the candidate are the same player")
    label = f"{args.model}_{args.baseline or 'reference'}_vs_{args.candidate or 'reference'}"
    out = args.out or Path("data/tournament") / f"versions_{label.replace(',', '-')}.jsonl"
    logs = out.with_suffix("")
    run = run_round_robin(
        [baseline, candidate],
        paired_openings(args.openings),
        args.base,
        args.increment,
        args.workers,
        out,
        log_dir=logs,
        max_depth=DEFAULT_MAX_DEPTH,
    )
    print(f"played {run['played']}, skipped {run['skipped']}, crashed {run['errors']}")
    summary = summarize(read_games(out), baseline)
    depths = mean_depths(logs, [baseline, candidate])
    lines = render(baseline, candidate, summary, depths, args.base, args.increment)
    report = args.report or Path("data/reports") / f"VERSIONS_{label.replace(',', '-')}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
