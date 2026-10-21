#!/usr/bin/env python3
"""Check a sample of the opening book against Stockfish, through the Lichess cloud evaluation.

python scripts/data/check_opening_book.py data/opening_book.json --fraction 0.02

A random share of the book entries (fixed seed) is judged by the public ``cloud-eval`` API of
Lichess (Stockfish). For each entry the script asks for the best lines of the position and takes the
evaluation of the book move (from the lines, or by evaluating the position it leads to). The move
passes when it is within ``--margin`` centipawns of the best line. Positions that the service does
not hold are counted apart and do not pass or fail. With ``--prune`` the failing entries are removed
and the book is written to ``--out``.

Writes ``data/reports/OPENING_BOOK_CHECK.md``.
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import chess

API = "https://lichess.org/api/cloud-eval"
#: Score of a forced mate, minus the moves it takes.
MATE_SCORE = 100_000

Fetch = Callable[[str, int], Optional[dict]]


def fetch_cloud_eval(fen: str, lines: int, pause: float = 1.0) -> Optional[dict]:
    """Ask the Lichess cloud evaluation for a position.

    :param fen: Position.
    :param lines: Number of best lines wanted.
    :param pause: Seconds waited before the request (the service limits the rate).
    :return: The JSON answer, or ``None`` when the service does not hold the position.
    """
    url = f"{API}?{urllib.parse.urlencode({'fen': fen, 'multiPv': lines})}"
    for _ in range(5):
        time.sleep(pause)
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if error.code == 429:
                time.sleep(60)
                continue
            raise
    raise RuntimeError("the cloud evaluation keeps refusing requests (HTTP 429)")


def line_score(line: dict) -> float:
    """Centipawns of a line from the point of view of the side to move.

    :param line: One entry of ``pvs``, with ``cp`` or ``mate``.
    :return: The score; a mate becomes a large number.
    """
    if "mate" in line:
        mate = line["mate"]
        return (1 if mate > 0 else -1) * (MATE_SCORE - abs(mate))
    return float(line["cp"])


def judge_entry(fen: str, uci: str, fetch: Fetch, lines: int = 5) -> Tuple[str, float]:
    """Compare a book move with the best line of the engine.

    :param fen: Book position (FEN or its first four fields).
    :param uci: Book move.
    :param fetch: Function ``(fen, lines) -> answer or None`` that queries the engine.
    :param lines: Best lines requested.
    :return: ``("ok", loss)`` with the centipawns lost against the best line, ``("missing", 0)``
        when the service does not hold the position, or ``("illegal", 0)``.
    """
    board = chess.Board(fen if len(fen.split()) >= 6 else f"{fen} 0 1")
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        return "illegal", 0.0
    answer = fetch(board.fen(), lines)
    if not answer or not answer.get("pvs"):
        return "missing", 0.0
    scores: Dict[str, float] = {}
    for line in answer["pvs"]:
        first = line["moves"].split()[0]
        scores[first] = line_score(line)
    best = max(scores.values())
    if uci in scores:
        return "ok", best - scores[uci]
    board.push(move)
    child = fetch(board.fen(), 1)
    if not child or not child.get("pvs"):
        return "missing", 0.0
    return "ok", best - (-line_score(child["pvs"][0]))


def sample_entries(book: Dict[str, str], fraction: float, seed: int) -> List[Tuple[str, str]]:
    """Random entries of the book.

    :param book: ``{position: move}``.
    :param fraction: Share of the entries to draw.
    :param seed: Seed of the draw.
    :return: The sampled ``(position, move)`` pairs, in a fixed order.
    """
    keys = sorted(book)
    count = max(1, round(len(keys) * fraction))
    chosen = random.Random(seed).sample(keys, min(count, len(keys)))
    return [(key, book[key]) for key in chosen]


def check_book(
    book: Dict[str, str], fraction: float, seed: int, margin: float, fetch: Fetch
) -> Dict[str, object]:
    """Judge a sample of the book.

    :param book: ``{position: move}``.
    :param fraction: Share of the entries to judge.
    :param seed: Seed of the draw.
    :param margin: Centipawns a move may lose against the best line and still pass.
    :param fetch: Function that queries the engine.
    :return: ``sampled``, ``judged``, ``missing``, ``illegal``, ``failed`` (list of
        ``(position, move, loss)``) and ``mean_loss``.
    """
    failed: List[Tuple[str, str, float]] = []
    losses: List[float] = []
    sampled = sample_entries(book, fraction, seed)
    missing = illegal = 0
    for fen, uci in sampled:
        status, loss = judge_entry(fen, uci, fetch)
        if status == "missing":
            missing += 1
        elif status == "illegal":
            illegal += 1
            failed.append((fen, uci, float("inf")))
        else:
            losses.append(loss)
            if loss > margin:
                failed.append((fen, uci, loss))
    return {
        "sampled": len(sampled),
        "judged": len(losses),
        "missing": missing,
        "illegal": illegal,
        "failed": failed,
        "mean_loss": sum(losses) / len(losses) if losses else 0.0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book", type=Path, help="book to check (JSON {fen: uci})")
    parser.add_argument("--fraction", type=float, default=0.02, help="share of entries judged")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=50.0, help="centipawns allowed to lose")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    parser.add_argument(
        "--prune", action="store_true", help="write the book without failing entries"
    )
    parser.add_argument("--out", type=Path, help="where --prune writes the book")
    parser.add_argument("--report", type=Path, default=Path("data/reports/OPENING_BOOK_CHECK.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Judge the sample and write the report.

    :param argv: Command-line arguments.
    :return: Process exit code.
    """
    args = parse_args(argv)
    book = json.loads(args.book.read_text(encoding="utf-8"))
    result = check_book(
        book,
        args.fraction,
        args.seed,
        args.margin,
        lambda fen, lines: fetch_cloud_eval(fen, lines, args.pause),
    )
    judged, failed = result["judged"], result["failed"]
    passed = (judged - len(failed) + result["illegal"]) / judged if judged else 0.0
    lines = [
        "# Opening book check against Stockfish (Lichess cloud evaluation)",
        "",
        f"Book `{args.book}`: {len(book):,} positions. Sample: {result['sampled']:,} entries "
        f"({args.fraction:.1%}, seed {args.seed}).",
        "",
        f"- Judged by the engine: {judged:,}; not held by the service: {result['missing']:,}",
        f"- Illegal moves: {result['illegal']}",
        f"- Within {args.margin:g} centipawns of the best line: {passed:.1%}",
        f"- Mean loss against the best line: {result['mean_loss']:.1f} centipawns",
        f"- Failing entries: {len(failed)}",
    ]
    for fen, uci, loss in failed[:20]:
        lines.append(f"  - `{fen}` `{uci}` loses {loss:.0f} cp")
    if args.prune:
        bad = {fen for fen, _, _ in failed}
        kept = {k: v for k, v in book.items() if " ".join(k.split()[:4]) not in bad}
        out = args.out or args.book
        out.write_text(json.dumps(kept, indent=0, sort_keys=True) + "\n", encoding="utf-8")
        lines.append(f"- Pruned book: {len(kept):,} positions -> {out}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
