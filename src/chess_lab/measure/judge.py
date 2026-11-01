"""Judge of the evaluation models against Stockfish.

Two kinds of agreement are measured on a test sample of :func:`~.sidecar.load_sample`:

* static accuracy: Spearman rank correlation and mean absolute error of the predicted expected
  score against the Stockfish expected score (Lichess win-probability scale);
* 1-ply best-move agreement: how often the move chosen by the model after one search step equals
  the Stockfish best move.
"""

import chess
import numpy as np
import pandas as pd

from chess_engine import _core
from chess_lab.models.representation import Representation, white_score


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks of ``values``, ties sharing their average rank.

    :param values: One-dimensional array.
    :return: Array of the same length with the ranks.
    """
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts).astype(np.float64)
    average = ends - (counts - 1) / 2.0
    return average[inverse]


def _spearman(pred: np.ndarray, truth: np.ndarray) -> float:
    """Spearman rank correlation: Pearson correlation of the average ranks.

    :return: The correlation, or NaN when undefined (fewer than two values or constant input).
    """
    if pred.size < 2:
        return float("nan")
    rank_pred, rank_truth = _average_ranks(pred), _average_ranks(truth)
    if rank_pred.std() == 0.0 or rank_truth.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(rank_pred, rank_truth)[0, 1])


def static_metrics(pred_white: np.ndarray, true_white: np.ndarray) -> dict:
    """Static accuracy of predicted expected scores.

    :param pred_white: Predicted expected scores for White, in ``[0, 1]``.
    :param true_white: Stockfish expected scores for White, in ``[0, 1]``.
    :return: ``{"spearman": ..., "mae": ..., "n": ...}``; mean absolute error in expected-score
        units, NaN for an empty sample.
    """
    pred = np.asarray(pred_white, dtype=np.float64)
    truth = np.asarray(true_white, dtype=np.float64)
    mae = float(np.mean(np.abs(pred - truth))) if pred.size else float("nan")
    return {"spearman": _spearman(pred, truth), "mae": mae, "n": int(pred.size)}


def one_ply_choice(evaluator: _core.Evaluator, board: chess.Board) -> chess.Move:
    """Best move of the mover after one ply, judged by the evaluation.

    A mating move ranks first; stalemate and other draws count as an even score.

    :param evaluator: Evaluation of the resulting positions, as the engine plays it.
    :param board: Position to move in; it is not modified.
    :return: The legal move with the highest evaluation for the side to move.
    :raises ValueError: If the position has no legal move.
    """
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("the position has no legal move")
    mover = board.turn
    scores = []
    for move in moves:
        child = board.copy(stack=False)
        child.push(move)
        if child.is_checkmate():
            scores.append(float("inf"))
        elif child.is_game_over():
            scores.append(0.5)
        else:
            scores.append(evaluator.evaluate(child.fen(), mover == chess.WHITE))
    return moves[max(range(len(moves)), key=lambda i: scores[i])]


def judge(evaluator: _core.Evaluator, test: pd.DataFrame, agreement_n: int = 20_000) -> dict:
    """Judge an evaluation on a test sample, on all positions and on each quiet slice.

    Static metrics use every row; the Stockfish truth is always on the Lichess win-probability
    scale. Best-move agreement uses the first ``agreement_n`` rows of the sample (rows without a
    best move are skipped) and is split by slice with the same rows.

    :param evaluator: Evaluation to judge, as the engine plays it (scores in ``[0.01, 0.99]``).
    :param test: Sample with columns ``fen``, ``cp``, ``mate``, ``best_move``, ``quiet``.
    :param agreement_n: Number of leading rows used for best-move agreement.
    :return: ``{"all": ..., "quiet": ..., "non_quiet": ...}``, each with the keys of
        :func:`static_metrics` plus ``best_move_agreement`` (NaN if no row) and
        ``agreement_n`` (rows with a best move that were compared).
    """
    fens = test["fen"].tolist()
    pred = np.array([evaluator.evaluate(fen, True) for fen in fens], dtype=np.float64)
    truth = white_score(test["cp"].to_numpy(), test["mate"].to_numpy(), Representation())
    quiet = test["quiet"].to_numpy(dtype=bool)

    head = min(agreement_n, len(fens))
    agree = np.empty(head, dtype=np.float64)
    for row in range(head):
        best = test["best_move"].iloc[row]
        if best is None or (isinstance(best, float) and np.isnan(best)):
            agree[row] = np.nan
            continue
        choice = one_ply_choice(evaluator, chess.Board(fens[row]))
        agree[row] = float(choice.uci() == best)

    def slice_report(mask: np.ndarray) -> dict:
        report = static_metrics(pred[mask], truth[mask])
        compared = agree[mask[:head]]
        compared = compared[~np.isnan(compared)]
        report["best_move_agreement"] = float(compared.mean()) if compared.size else float("nan")
        report["agreement_n"] = int(compared.size)
        return report

    everything = np.ones(len(fens), dtype=bool)
    return {
        "all": slice_report(everything),
        "quiet": slice_report(quiet),
        "non_quiet": slice_report(~quiet),
    }
