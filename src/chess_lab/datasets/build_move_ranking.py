#!/usr/bin/env python3
"""Build the move-ranking test set: every legal move of test positions, scored by Stockfish.

The positions are the first rows of the seeded test sample used by ``measure.evaluate``. Stockfish
analyses each one with one line per legal move (MultiPV) at a fixed depth, so every move has a score
from the mover's point of view. The result has one row per (position, move); positions with fewer
than two legal moves or already finished are skipped.

    python -m chess_lab.datasets.build_move_ranking --engine ~/stockfish/stockfish --rows 5000 --depth 16 --jobs 7
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.engine
import pandas as pd

from chess_engine.paths import ROOT
from chess_lab.datasets.sidecar import load_sample
from chess_lab.datasets.split import TEST
from chess_lab.models.defaults import RAW_DIR, SAMPLE_SEED, SIDECAR_DIR

#: Rows of the test sample the positions are drawn from (the same sample as ``measure.evaluate``).
SAMPLE_ROWS = 200_000
#: Default output file.
OUT = ROOT / "data" / "LichessPositions" / "move_ranking" / "test.parquet"


def label_position(
    engine: chess.engine.SimpleEngine, fen: str, depth: int
) -> list[tuple[str, float, float]]:
    """Stockfish score of every legal move of a position.

    :param engine: Open UCI engine.
    :param fen: Position to analyse.
    :param depth: Fixed search depth.
    :return: One ``(move, cp, mate)`` per legal move, from the mover's point of view; ``cp`` is NaN on
        a mate score and ``mate`` is NaN otherwise (positive when the mover mates).
    """
    board = chess.Board(fen)
    count = board.legal_moves.count()
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=count)
    rows = []
    for info in infos:
        score = info["score"].pov(board.turn)
        mate = score.mate()
        rows.append(
            (
                info["pv"][0].uci(),
                float("nan") if mate is not None else float(score.score()),
                float("nan") if mate is None else float(mate),
            )
        )
    return rows


def _label_chunk(args: tuple[str, int, list[str]]) -> list[tuple[str, str, float, float]]:
    """Worker: label a list of positions with its own engine."""
    engine_path, depth, fens = args
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        engine.configure({"Threads": 1, "Hash": 64})
        return [(fen, *row) for fen in fens for row in label_position(engine, fen, depth)]
    finally:
        engine.quit()


def eligible(fens: list[str], rows: int) -> list[str]:
    """The first ``rows`` positions that are unfinished and have at least two legal moves."""
    kept = []
    for fen in fens:
        board = chess.Board(fen)
        if not board.is_game_over() and board.legal_moves.count() >= 2:
            kept.append(fen)
        if len(kept) == rows:
            break
    return kept


def main() -> None:
    """Parse the arguments, label the positions and write the parquet file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", required=True, help="path of a UCI engine (Stockfish)")
    parser.add_argument("--rows", type=int, default=5000, help="positions to label")
    parser.add_argument("--depth", type=int, default=16, help="fixed Stockfish depth")
    parser.add_argument("--jobs", type=int, default=4, help="engine processes")
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--sidecar", type=Path, default=SIDECAR_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    sample = load_sample(args.raw, args.sidecar, TEST, SAMPLE_ROWS, seed=SAMPLE_SEED)
    quiet = dict(zip(sample["fen"], sample["quiet"], strict=True))
    fens = eligible(sample["fen"].tolist(), args.rows)
    chunks = [(args.engine, args.depth, fens[i :: args.jobs * 8]) for i in range(args.jobs * 8)]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = [row for part in pool.map(_label_chunk, chunks) for row in part]
    table = pd.DataFrame(rows, columns=["fen", "move", "cp", "mate"])
    table["quiet"] = table["fen"].map(quiet).astype(bool)
    table = table.sort_values(["fen", "move"], kind="stable", ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(f"{len(fens)} positions, {len(table)} moves, depth {args.depth} -> {args.out}")


if __name__ == "__main__":
    main()
