#!/usr/bin/env python3
"""Compare the rules that predict the duration of the next iterative-deepening iteration.

python scripts/bench/compare_time_estimators.py data/reports/iteration_times.jsonl

The input is the output of ``record_iteration_times.py``: for each move the duration and node count
of every iteration, measured without any stopping rule. The script replays the deepening loop on
those series with each estimator of ``chess_engine.search.time_estimators`` and several time
budgets, and reports what each rule would have done:

* depth: mean completed depth of the move;
* seconds: mean time used, as a share of the budget;
* regret: mean number of plies missed against an oracle that knows every duration (the deepest
  iteration that ends within the budget plus the 15 % overshoot);
* premature: share of moves that stopped although the next iteration would have fitted;
* cut: share of moves where an iteration was started and interrupted by the hard limit, so its
  time was wasted.

Writes ``data/reports/TIME_ESTIMATORS.md``.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from chess_engine.search.time_estimators import ESTIMATORS, Estimator

#: The numbers of ``TimeManager`` and of the iterative deepening loop.
OVERSHOOT = 1.15
HARD_EXTENSION, HARD_MARGIN = 1.5, 0.3
MIN_DEPTH = 3
#: Budgets (seconds) replayed.
BUDGETS = (2.0, 5.0, 10.0, 20.0, 40.0)

Series = List[Tuple[int, float, int, bool]]


def hard_limit(budget: float) -> float:
    """Elapsed time after which a running iteration is interrupted."""
    return HARD_EXTENSION * budget - HARD_MARGIN


def oracle_depth(series: Series, budget: float) -> int:
    """Deepest completed iteration whose cumulative time fits in the budget plus the overshoot."""
    elapsed, best = 0.0, 0
    for depth, seconds, _, truncated in series:
        elapsed += seconds
        if truncated or elapsed > budget * OVERSHOOT:
            break
        best = depth
    return best


def simulate(series: Series, estimator: Estimator, budget: float) -> Dict[str, Any]:
    """Replay the deepening loop on one recorded series.

    :param series: ``(depth, seconds, nodes, truncated)`` per iteration, in order.
    :param estimator: Rule that predicts the next duration.
    :param budget: Soft budget in seconds.
    :return: ``depth`` completed, ``seconds`` used, ``cut`` (an iteration was interrupted) and
        ``premature`` (the loop stopped although the next iteration would have fitted).
    """
    hard = hard_limit(budget)
    elapsed, completed, history = 0.0, 0, []
    cut = premature = False
    for depth, seconds, nodes, truncated in series:
        if depth > MIN_DEPTH and history:
            predicted = estimator(history)
            if predicted > (budget - elapsed) * OVERSHOOT:
                # it would have finished within the budget plus the overshoot
                premature = not truncated and elapsed + seconds <= budget * OVERSHOOT
                break
        if truncated or elapsed + seconds > hard:
            elapsed, cut = hard, True
            break
        elapsed += seconds
        history.append((seconds, nodes))
        completed = depth
        if depth >= MIN_DEPTH and elapsed >= budget:
            break
    return {"depth": completed, "seconds": elapsed, "cut": cut, "premature": premature}


def evaluate(all_series: Sequence[Series], estimator: Estimator, budget: float) -> Dict[str, float]:
    """Average the outcome of a rule over all recorded moves.

    :param all_series: One series per move.
    :param estimator: Rule that predicts the next duration.
    :param budget: Soft budget in seconds.
    :return: ``depth``, ``seconds_share``, ``regret``, ``premature`` and ``cut``.
    :raises ValueError: If there are no series.
    """
    if not all_series:
        raise ValueError("no series to evaluate")
    outcomes = [simulate(s, estimator, budget) for s in all_series]
    n = len(outcomes)
    return {
        "depth": sum(o["depth"] for o in outcomes) / n,
        "seconds_share": sum(o["seconds"] for o in outcomes) / n / budget,
        "regret": sum(
            oracle_depth(s, budget) - o["depth"] for s, o in zip(all_series, outcomes, strict=True)
        )
        / n,
        "premature": sum(o["premature"] for o in outcomes) / n,
        "cut": sum(o["cut"] for o in outcomes) / n,
    }


def load_series(path: Path) -> List[Series]:
    """Read the recorded moves.

    :param path: JSON-lines file written by ``record_iteration_times.py``.
    :return: One series per move that has at least four iterations.
    """
    series = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        its = [(int(d), float(s), int(n), bool(t)) for d, s, n, t in record["iterations"]]
        if len(its) >= MIN_DEPTH + 1:
            series.append(its)
    return series


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", type=Path, help="output of record_iteration_times.py")
    parser.add_argument("--report", type=Path, default=Path("data/reports/TIME_ESTIMATORS.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Replay every estimator and write the report.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    series = load_series(args.series)
    if not series:
        print(f"no usable series in {args.series}")
        return 1
    lines = [
        "# Estimators of the next iteration, replayed on recorded games",
        "",
        f"{len(series)} moves. depth = mean completed depth; time = mean share of the budget used; "
        "regret = plies missed against an oracle; premature = stopped although the next iteration "
        "would have fitted; cut = an iteration was started and interrupted by the hard limit.",
    ]
    for budget in BUDGETS:
        lines += [
            "",
            f"## Budget {budget:g} s",
            "",
            "| estimator | depth | time | regret | premature | cut |",
            "|---|---|---|---|---|---|",
        ]
        for name, estimator in ESTIMATORS.items():
            if name == "never_stop":
                continue
            r = evaluate(series, estimator, budget)
            lines.append(
                f"| {name} | {r['depth']:.2f} | {r['seconds_share']:.0%} | {r['regret']:.2f} | "
                f"{r['premature']:.0%} | {r['cut']:.0%} |"
            )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
