#!/usr/bin/env python3
"""Build the game-position test set: positions of engine games, scored by Stockfish.

The Lichess test sample holds positions people reached; the engine evaluates the positions its own
search reaches. This set draws them from the fixed-node screening games of the round robin
(fastchess PGN files, which start from ``openings.epd`` and use no opening book): up to
``--per-game`` positions per game, chosen at random among those reached after the starting
position, skipping finished positions and repeats. Stockfish scores each one at a fixed depth from
White's point of view, and its best move gives the quiet flag, as in the sidecars.

    python -m chess_lab.datasets.build_game_positions --engine /usr/games/stockfish --depth 14 --jobs 6
"""

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd

from chess_engine.paths import ROOT
from chess_lab.datasets.book import fen_key
from chess_lab.datasets.stockfish import cp_mate, label_parallel
from chess_lab.models.target import is_quiet

#: Default PGN files: the fixed-node screenings fetched from the node.
GAMES = ROOT / "data" / "results" / "roundrobin"
GAMES_GLOB = "nodes_*/match.pgn"
#: Default output file.
OUT = ROOT / "data" / "results" / "game_positions.parquet"
#: Seed of the choice of positions within each game.
SEED = 0

Row = Tuple[str, float, float, Optional[str]]


def game_positions(pgn: Path, per_game: int, rng: np.random.Generator) -> List[str]:
    """Up to ``per_game`` unfinished positions of every game of a PGN file, as FENs.

    :param pgn: PGN file.
    :param per_game: Maximum positions taken from one game.
    :param rng: Random generator of the choice within each game.
    :return: The FENs, game after game.
    """
    fens: List[str] = []
    with open(pgn, encoding="utf-8") as handle:
        while (game := chess.pgn.read_game(handle)) is not None:
            board = game.board()
            reached = []
            for move in game.mainline_moves():
                board.push(move)
                if not board.is_game_over():
                    reached.append(board.fen())
            if reached:
                take = min(per_game, len(reached))
                fens.extend(
                    reached[i] for i in sorted(rng.choice(len(reached), take, replace=False))
                )
    return fens


def unique(fens: Iterable[str]) -> List[str]:
    """The FENs in their first order, each once (counters ignored: same position, same score).

    :param fens: FENs.
    :return: The first FEN of every position.
    """
    seen, kept = set(), []
    for fen in fens:
        key = fen_key(fen)
        if key not in seen:
            seen.add(key)
            kept.append(fen)
    return kept


def label_position(engine: chess.engine.SimpleEngine, fen: str, depth: int) -> Row:
    """Stockfish score and best move of a position.

    :param engine: Open UCI engine.
    :param fen: Position to analyse.
    :param depth: Fixed search depth.
    :return: ``(fen, cp, mate, best_move)`` from White's point of view; ``cp`` is NaN on a mate
        score and ``mate`` is NaN otherwise (positive when White mates); ``best_move`` is None
        when the engine gives no line.
    """
    info = engine.analyse(chess.Board(fen), chess.engine.Limit(depth=depth))
    cp, mate = cp_mate(info["score"].white())
    pv = info.get("pv")
    return fen, cp, mate, pv[0].uci() if pv else None


def table_of(rows: List[Row]) -> pd.DataFrame:
    """The labelled positions as the columns ``judge`` reads, plus the quiet flag.

    :param rows: ``(fen, cp, mate, best_move)`` per position.
    :return: Columns ``fen``, ``cp``, ``mate``, ``best_move``, ``quiet``.
    """
    table = pd.DataFrame(rows, columns=["fen", "cp", "mate", "best_move"])
    table["quiet"] = [
        is_quiet(chess.Board(fen), best)
        for fen, best in zip(table["fen"], table["best_move"], strict=True)
    ]
    return table


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options.

    :param argv: Arguments; None reads ``sys.argv``.
    :return: The parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", required=True, help="path of a UCI engine (Stockfish)")
    parser.add_argument("--depth", type=int, default=14, help="fixed Stockfish depth")
    parser.add_argument("--per-game", type=int, default=4, help="maximum positions per game")
    parser.add_argument("--jobs", type=int, default=4, help="engine processes")
    parser.add_argument(
        "--pgn", type=Path, nargs="*", help=f"PGN files (default: {GAMES}/{GAMES_GLOB})"
    )
    parser.add_argument("--limit", type=int, help="label only the first N positions (dry run)")
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Draw the positions, label them and write the parquet file.

    :param argv: Arguments; None reads ``sys.argv``.
    :return: Exit code.
    """
    args = parse_args(argv)
    pgns = args.pgn or sorted(GAMES.glob(GAMES_GLOB))
    rng = np.random.default_rng(SEED)
    fens = unique(fen for pgn in pgns for fen in game_positions(pgn, args.per_game, rng))
    fens = fens[: args.limit] if args.limit else fens
    rows = [
        row for _, row in label_parallel(args.engine, args.depth, fens, args.jobs, label_position)
    ]
    table = table_of(rows).sort_values("fen", kind="stable", ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(
        f"{len(pgns)} PGN files, {len(table)} positions ({int(table['quiet'].sum())} quiet), "
        f"depth {args.depth} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
