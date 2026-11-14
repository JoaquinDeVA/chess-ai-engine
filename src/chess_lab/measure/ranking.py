"""Move-ordering judge: how well a model ranks the legal moves of one position.

For each position of the move-ranking test set (:mod:`~chess_lab.datasets.build_move_ranking`) the
model scores the position after every legal move, as the engine does at one ply, and the order is
compared with Stockfish's scores of the same moves:

* ``tau``: Kendall tau-b between the model's and Stockfish's scores of the moves of one position,
  averaged over the positions where Stockfish does not score all moves alike;
* ``top1`` and ``top3``: share of positions where the model's first move (or one of its first
  three) is a move Stockfish scores best.

Positions are split by the quiet flag of the sidecar, as in :mod:`~chess_lab.measure.judge`.
"""

import chess
import numpy as np
import pandas as pd

from chess_lab.models.target import Representation, white_score

#: Score of a mating move: above any evaluation.
_MATE_SCORE = 2.0


def kendall_tau(pred: np.ndarray, truth: np.ndarray) -> float:
    """Kendall tau-b between two score vectors.

    :param pred: Predicted scores.
    :param truth: True scores, same length.
    :return: The correlation, or NaN when either vector is constant or has fewer than two values.
    """
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    sign_pred = np.sign(pred[:, None] - pred[None, :])
    sign_truth = np.sign(truth[:, None] - truth[None, :])
    pairs_pred, pairs_truth = np.abs(sign_pred).sum(), np.abs(sign_truth).sum()
    if pairs_pred == 0 or pairs_truth == 0:
        return float("nan")
    return float((sign_pred * sign_truth).sum() / np.sqrt(pairs_pred * pairs_truth))


def move_scores(evaluator, board: chess.Board, moves: list[chess.Move]) -> np.ndarray:
    """Score of each move for the mover: the evaluation of the position after it.

    A mating move scores above everything; stalemate and other draws count as ``0.5``.

    :param evaluator: Evaluation as the engine plays it.
    :param board: Position to move in; it is not modified.
    :param moves: Legal moves to score.
    :return: One score per move.
    """
    mover = board.turn
    scores = []
    for move in moves:
        child = board.copy(stack=False)
        child.push(move)
        if child.is_checkmate():
            scores.append(_MATE_SCORE)
        elif child.is_game_over():
            scores.append(0.5)
        else:
            scores.append(evaluator.evaluate(child.fen(), mover == chess.WHITE))
    return np.array(scores, dtype=np.float64)


def judge_ranking(evaluator, table: pd.DataFrame) -> dict:
    """Judge the move ordering of an evaluation on the move-ranking test set.

    :param evaluator: Evaluation to judge, as the engine plays it.
    :param table: Rows ``fen``, ``move``, ``cp``, ``mate`` (mover's point of view), ``quiet``.
    :return: See :func:`judge_scorer`.
    """
    return judge_scorer(lambda fen, moves: move_scores(evaluator, chess.Board(fen), moves), table)


def judge_scorer(scorer, table: pd.DataFrame) -> dict:
    """Judge any move scorer on the move-ranking test set.

    :param scorer: Function ``(fen, moves) -> scores``, one score per move, higher is better.
    :param table: Rows ``fen``, ``move``, ``cp``, ``mate`` (mover's point of view), ``quiet``.
    :return: ``{"all": ..., "quiet": ..., "non_quiet": ...}``, each with ``tau``, ``top1``,
        ``top3`` and ``positions``.
    """
    truth_all = white_score(table["cp"].to_numpy(), table["mate"].to_numpy(), Representation())
    table = table.assign(truth=truth_all)
    taus, top1, top3, quiet = [], [], [], []
    for fen, group in table.groupby("fen", sort=False):
        moves = [chess.Move.from_uci(m) for m in group["move"]]
        truth = group["truth"].to_numpy()
        pred = scorer(fen, moves)
        best = truth == truth.max()
        order = np.argsort(-pred, kind="stable")
        taus.append(kendall_tau(pred, truth))
        top1.append(float(best[order[0]]))
        top3.append(float(best[order[:3]].any()))
        quiet.append(bool(group["quiet"].iloc[0]))
    taus, top1, top3 = np.array(taus), np.array(top1), np.array(top3)
    quiet = np.array(quiet, dtype=bool)

    def report(mask: np.ndarray) -> dict:
        defined = taus[mask][~np.isnan(taus[mask])]
        return {
            "tau": float(defined.mean()) if defined.size else float("nan"),
            "top1": float(top1[mask].mean()) if mask.any() else float("nan"),
            "top3": float(top3[mask].mean()) if mask.any() else float("nan"),
            "positions": int(mask.sum()),
        }

    return {
        "all": report(np.ones(len(taus), dtype=bool)),
        "quiet": report(quiet),
        "non_quiet": report(~quiet),
    }


def main() -> int:
    """Judge Stockfish at a low depth as a reference on the move-ranking test set."""
    import argparse
    from pathlib import Path

    import chess.engine

    from chess_lab.datasets.build_move_ranking import OUT, label_position

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--engine", required=True, help="path of Stockfish")
    parser.add_argument("--depth", type=int, default=1, help="search depth of the reference")
    parser.add_argument("--ranking", type=Path, default=OUT, help="move-ranking test set")
    args = parser.parse_args()
    table = pd.read_parquet(args.ranking)
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    try:
        engine.configure({"Threads": 1, "Hash": 64})

        def scorer(fen: str, moves: list[chess.Move]) -> np.ndarray:
            rows = {m: (cp, mate) for m, cp, mate in label_position(engine, fen, args.depth)}
            cp = np.array([rows[m.uci()][0] for m in moves])
            mate = np.array([rows[m.uci()][1] for m in moves])
            return white_score(cp, mate, Representation())

        report = judge_scorer(scorer, table)
    finally:
        engine.quit()
    for name, part in report.items():
        print(
            f"stockfish depth {args.depth} ({name}, {part['positions']} positions): "
            f"tau {part['tau']:.4f}, top1 {part['top1']:.4f}, top3 {part['top3']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
