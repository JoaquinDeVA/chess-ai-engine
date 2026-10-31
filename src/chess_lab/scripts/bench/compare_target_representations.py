#!/usr/bin/env python3
"""Representation bake-off: which target, point of view and training subset suit linear_pst.

python src/chess_lab/scripts/bench/compare_target_representations.py --train-positions 5000000 --test-positions 200000 \
    --effort 30 --tiebreak-games 200 --out data/reports/bakeoff.json

Every variant changes one axis of the default representation and is trained as a tagged model
``linear_pst@<prefix>-<index>`` (artifact and card; the prefix is ``bake`` unless
``--tag-prefix`` says otherwise). The variants are judged on the same test sample
with :func:`~chess_lab.comparison.judge.judge`. The two best by ``all`` best-move agreement then
play a round-robin at equal time per move. The script stops there: it writes the report, which
ends with ``Decision: pending``, and deletes the tagged models it created unless ``--keep-models`` is given. The script refuses
to start if one of its tagged models or cards already exists, so it never overwrites or deletes
models it did not create. A hard kill can leave ``@<prefix>-*`` files behind; they are safe to
delete by hand.

The variants that filter on ``depth`` or ``mate`` drop rows after loading, on the training stream
and on the validation sample. The test sample always keeps every row.
"""

import argparse
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from chess_engine.models.cards import card_path, current_commit, today, update_card
from chess_engine.models.linear_eval import LinearEvalModel
from chess_engine.models.registry import artifact_path, get_model_class
from chess_lab.comparison.judge import judge
from chess_lab.comparison.standings import record_cards, summarize
from chess_lab.comparison.tournament import paired_openings, read_games, run_round_robin
from chess_lab.models import fit_stream, new_model
from chess_lab.models.linear_eval import DEFAULT_FIT_CONFIG
from chess_lab.models.representation import Representation
from chess_lab.models.spec import SAMPLE_SEED
from chess_lab.training.dataset import TEST, TRAIN, VALIDATION
from chess_lab.training.effort import effort_config, fit_config
from chess_lab.training.sidecar import iter_batches, load_sample
from chess_lab.training.train import (
    DEFAULT_RAW_DIR,
    DEFAULT_SIDECAR_DIR,
    MAX_VALIDATION_ROWS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

MODEL = "linear_pst"


@dataclass(frozen=True)
class Variant:
    """One arm of the bake-off: a representation and the rows dropped after loading.

    :param label: Name shown in the report.
    :param rep: Representation the model is trained with.
    :param min_depth: Keep only the rows whose raw search depth is at least this (None: all).
    :param drop_mates: Drop the rows with a mate score.
    """

    label: str
    rep: Representation
    min_depth: Optional[int] = None
    drop_mates: bool = False

    def keeps(self, frame: pd.DataFrame) -> np.ndarray:
        """Rows of a frame (with ``depth`` and ``mate`` columns) that the variant keeps.

        :param frame: Rows of a sample or of a training batch.
        :return: Boolean array, one entry per row.
        """
        keep = np.ones(len(frame), dtype=bool)
        if self.min_depth is not None:
            keep &= (frame["depth"] >= self.min_depth).to_numpy()
        if self.drop_mates:
            keep &= frame["mate"].isna().to_numpy()
        return keep


#: The eight arms; the first is the default representation.
VARIANTS = (
    Variant("winprob k=0.00368 stm all", Representation()),
    Variant("cp_linear stm all", Representation(scale="cp_linear")),
    Variant("winprob k=0.0025 stm all", Representation(k=0.0025)),
    Variant("winprob k=0.005 stm all", Representation(k=0.005)),
    Variant("winprob k=0.00368 white all", Representation(perspective="white")),
    Variant("winprob k=0.00368 stm quiet", Representation(filter="quiet")),
    Variant("winprob k=0.00368 stm all, depth >= 20", Representation(), min_depth=20),
    Variant("winprob k=0.00368 stm all, mates dropped", Representation(), drop_mates=True),
)


class _Passes:
    """Re-iterable training stream: every iteration starts a new pass, as ``fit_stream`` needs."""

    def __init__(self, make: Callable[[], Iterator[pd.DataFrame]]):
        self._make = make

    def __iter__(self) -> Iterator[pd.DataFrame]:
        return self._make()


def training_batches(
    variant: Variant,
    raw_dir: Path,
    sidecar_dir: Path,
    positions: int,
    batch_size: int,
    quiet: bool,
    mate_keep: float = 1.0,
) -> Iterator[pd.DataFrame]:
    """Shuffled TRAIN batches of a variant, filtered after loading, ``positions`` rows in total.

    Filtered batches are smaller than ``batch_size``; the rows are the same as those of
    :func:`~chess_lab.training.sidecar.iter_batches` with the filtered rows removed.

    :param variant: The arm whose filter applies.
    :param raw_dir: Directory with the raw shards.
    :param sidecar_dir: Directory with their sidecars.
    :param positions: Kept rows to yield in one pass.
    :param batch_size: Rows per batch before filtering.
    :param quiet: Yield only quiet positions.
    :param mate_keep: Share of the mate positions yielded (as in ``fit``); 1 keeps them all.
    :return: Iterator of DataFrames with the columns of the sampler.
    """
    seen = 0
    stream = iter_batches(
        raw_dir,
        sidecar_dir,
        TRAIN,
        batch_size,
        None,
        SAMPLE_SEED,
        quiet_only=quiet,
        mate_keep=mate_keep,
    )
    for frame in stream:
        frame = frame[variant.keeps(frame)].iloc[: positions - seen]
        seen += len(frame)
        if len(frame):
            yield frame.reset_index(drop=True)
        if seen >= positions:
            return


def _agreement(result: Dict[str, Any]) -> float:
    """``all`` best-move agreement of a result; NaN counts as the worst."""
    value = result["judge"]["all"]["best_move_agreement"]
    return -math.inf if math.isnan(value) else value


def _cell(value: Optional[float], digits: int = 3) -> str:
    """Format a number for a table cell; ``—`` when there is none."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def render_markdown(
    results: Sequence[Dict[str, Any]], tiebreak: Dict[str, Any], meta: Dict[str, Any]
) -> str:
    """Report of the bake-off: the variant table, the tie-break and the pending decision.

    :param results: One result per variant (see :func:`main`).
    :param tiebreak: Players, time control and summary of the tie-break.
    :param meta: Free-form facts printed above the table.
    :return: Markdown text, ending with ``Decision: pending``.
    """
    lines = [
        "# Representation bake-off (linear_pst)",
        "",
        *(f"- {key}: {value}" for key, value in meta.items()),
        "",
        "Each variant changes one axis of the default representation. Judged on the same test "
        "sample: Spearman and MAE of the expected score against Stockfish, and 1-ply best-move "
        "agreement, for all positions and for the quiet and non-quiet slices. Training seconds "
        "are the fit of the variant (featurization and validation included).",
        "",
        "| # | Variant | Train rows | Training s | Spearman all | MAE all | Agreement all "
        "| Spearman quiet | MAE quiet | Agreement quiet | Spearman non-quiet | MAE non-quiet "
        "| Agreement non-quiet |",
        "|---" * 13 + "|",
    ]
    for result in results:
        j = result["judge"]
        lines.append(
            f"| {result['index']} | {result['label']} | {result['train_rows']} | "
            f"{_cell(result['seconds'], 1)} | "
            f"{_cell(j['all']['spearman'])} | {_cell(j['all']['mae'])} | "
            f"{_cell(j['all']['best_move_agreement'])} | "
            f"{_cell(j['quiet']['spearman'])} | {_cell(j['quiet']['mae'])} | "
            f"{_cell(j['quiet']['best_move_agreement'])} | "
            f"{_cell(j['non_quiet']['spearman'])} | {_cell(j['non_quiet']['mae'])} | "
            f"{_cell(j['non_quiet']['best_move_agreement'])} |"
        )
    anchor = tiebreak["players"][0]
    lines += [
        "",
        "## Tie-break",
        "",
        "The two variants with the best `all` best-move agreement play a round-robin at equal "
        f"time per move ({tiebreak['time_control']}, {tiebreak['openings']} openings, "
        f"{tiebreak['games']} games). Elo is relative to `{anchor}` = 0.",
        "",
        "| Player | Games | W-D-L | Score % | Elo vs anchor |",
        "|---|---|---|---|---|",
    ]
    for spec in tiebreak["players"]:
        p = tiebreak["summary"][spec]
        elo = "—" if p["elo_vs_anchor"] is None else f"{p['elo_vs_anchor']:+.0f}"
        score = "—" if p["score"] is None else f"{p['score']:.1f}"
        lines.append(
            f"| {spec} | {p['games']} | {p['wins']}-{p['draws']}-{p['losses']} | {score} | {elo} |"
        )
    lines += ["", "Decision: pending"]
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR, help="raw shards directory")
    parser.add_argument(
        "--sidecar", type=Path, default=DEFAULT_SIDECAR_DIR, help="sidecar directory"
    )
    parser.add_argument("--train-positions", type=int, default=5_000_000)
    parser.add_argument("--test-positions", type=int, default=200_000)
    parser.add_argument("--effort", type=int, default=30, help="epochs and patience, 0-100")
    parser.add_argument("--tiebreak-games", type=int, default=200, help="openings per pair")
    parser.add_argument("--base", type=float, default=60.0, help="tie-break clock, seconds")
    parser.add_argument("--increment", type=float, default=1.0, help="seconds per move")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="parallel tie-break games; capped at cores - 1",
    )
    parser.add_argument(
        "--tag-prefix",
        default="bake",
        help="tagged models are linear_pst@<prefix>-<index> (letters, digits, '.', '_', '-')",
    )
    parser.add_argument("--keep-models", action="store_true", help="keep the tagged models")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/reports/bakeoff.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "data/reports/BAKEOFF.md")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.tag_prefix):
        parser.error("--tag-prefix must contain only letters, digits, '.', '_' or '-'")
    if min(args.train_positions, args.test_positions, args.tiebreak_games) < 1:
        parser.error("--train-positions, --test-positions and --tiebreak-games must be >= 1")
    try:
        effort_config(args.effort, args.train_positions)
    except ValueError as error:
        parser.error(str(error))
    return args


def train_variant(
    variant: Variant, args: argparse.Namespace, plan: Dict[str, int], validation: pd.DataFrame
) -> Tuple[LinearEvalModel, Dict[str, Any]]:
    """Fit one variant on the TRAIN split, with the epochs of ``plan`` and default l2 and lr.

    :param variant: The arm to train.
    :param args: Parsed arguments (``raw``, ``sidecar``, ``train_positions``).
    :param plan: Output of :func:`~chess_lab.training.effort.effort_config`.
    :param validation: Validation rows of the variant, for early stopping.
    :return: The fitted model and the summary of ``fit_stream``.
    """
    quiet = variant.rep.filter == "quiet"
    config = fit_config(plan, DEFAULT_FIT_CONFIG["l2"], DEFAULT_FIT_CONFIG["lr"])
    batches = _Passes(
        lambda: training_batches(
            variant,
            args.raw,
            args.sidecar,
            args.train_positions,
            config["batch_size"],
            quiet,
            mate_keep=variant.rep.mate_keep,
        )
    )
    model = new_model(get_model_class(MODEL), variant.rep)
    return model, fit_stream(model, batches, validation, config)


def main(argv: Optional[List[str]] = None) -> int:
    """Train and judge the variants, run the tie-break, and write the report.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    if not sorted(args.raw.glob("part-*.parquet")):
        print(f"no raw shards in {args.raw}")
        return 2
    specs = [f"{MODEL}@{args.tag_prefix}-{index}" for index in range(len(VARIANTS))]
    taken = [spec for spec in specs if artifact_path(spec).exists() or card_path(spec).exists()]
    if taken:
        print(
            f"refusing to start: {', '.join(taken)} already exist (artifact or card). "
            "Use another --tag-prefix, or delete them by hand."
        )
        return 2
    model_class = get_model_class(MODEL)
    plan = effort_config(args.effort, args.train_positions)

    print("Loading the validation and test samples ...")
    # Validation samples keyed by quiet_only: the quiet variant early-stops on quiet positions.
    validation = {
        quiet: load_sample(
            args.raw, args.sidecar, VALIDATION, MAX_VALIDATION_ROWS, SAMPLE_SEED, quiet_only=quiet
        )
        for quiet in (False, True)
    }
    test = load_sample(args.raw, args.sidecar, TEST, args.test_positions, SAMPLE_SEED)
    print(f"{len(test):,} test rows")

    results: List[Dict[str, Any]] = []
    created: List[str] = []
    try:
        for index, variant in enumerate(VARIANTS):
            spec = specs[index]
            created.append(spec)
            print(f"[{index}] {variant.label}")
            sample = validation[variant.rep.filter == "quiet"]
            sample = sample[variant.keeps(sample)]
            model, summary = train_variant(variant, args, plan, sample)
            model.save(artifact_path(spec))
            update_card(
                spec,
                status="bake-off",
                architecture=model_class.architecture,
                features=model_class.feature_set,
                parameters=model.count_parameters(),
                size_kb=round(artifact_path(spec).stat().st_size / 1024, 1),
                training={
                    "dataset": str(args.raw),
                    "date": today(),
                    "commit": current_commit(),
                    "variant": variant.label,
                    "min_depth": variant.min_depth,
                    "drop_mates": variant.drop_mates,
                    **summary,
                    "train_samples": summary["train_rows"],
                    "effort": args.effort,
                    "effort_config": plan,
                    "representation": variant.rep.to_dict(),
                },
            )
            judged = judge(model, test)
            results.append(
                {
                    "index": index,
                    "spec": spec,
                    "label": variant.label,
                    "representation": variant.rep.to_dict(),
                    "min_depth": variant.min_depth,
                    "drop_mates": variant.drop_mates,
                    "train_rows": summary["train_rows"],
                    "seconds": summary["seconds"],
                    "judge": judged,
                }
            )
            print(
                f"    all-positions best-move agreement {judged['all']['best_move_agreement']:.3f}"
            )

        ranked = sorted(results, key=lambda r: (-_agreement(r), r["index"]))
        pair = [ranked[0]["spec"], ranked[1]["spec"]]
        print(f"Tie-break between {pair[0]} (anchor) and {pair[1]} ...")
        tiebreak = play_tiebreak(pair, args)
        recorded = record_cards(
            {"players": tiebreak["summary"]}, pair, pair[0], args.base, args.increment
        )
        if recorded:
            print("Model cards updated: " + ", ".join(recorded))
    finally:
        if not args.keep_models:
            for spec in created:
                artifact_path(spec).unlink(missing_ok=True)
                card_path(spec).unlink(missing_ok=True)

    meta = {
        "date": today(),
        "commit": current_commit(),
        "training positions per variant": f"{args.train_positions:,}",
        "test positions": f"{len(test):,}",
        "effort": args.effort,
        "epochs": plan["max_epochs"],
        "validation rows (before the variant filters)": f"{len(validation[False]):,}",
        "mate_keep": 1.0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "settings": {key: str(value) for key, value in meta.items()},
                "variants": results,
                "tiebreak": tiebreak,
            },
            indent=2,
        )
        + "\n"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(results, tiebreak, meta))
    print(f"Wrote {args.out} and {args.report}. Decision: pending.")
    return 0


def play_tiebreak(pair: List[str], args: argparse.Namespace) -> Dict[str, Any]:
    """Round-robin between two players at equal time per move, in a temporary file.

    :param pair: The two specs; the first is the Elo anchor.
    :param args: Parsed arguments (``tiebreak_games``, ``base``, ``increment``, ``workers``).
    :return: Players, time control, openings, games and the summary of :func:`summarize`.
    """
    openings = paired_openings(args.tiebreak_games)
    with tempfile.TemporaryDirectory() as tmp:
        games = Path(tmp) / "tiebreak.jsonl"
        run = run_round_robin(pair, openings, args.base, args.increment, args.workers, games)
        summary = summarize(read_games(games), pair[0])
    return {
        "players": pair,
        "anchor": pair[0],
        "time_control": f"{args.base:g}+{args.increment:g} s",
        "openings": len(openings),
        "games": run["games"],
        "errors": run["errors"],
        "summary": summary["players"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
