"""Judge of the evaluation models against Stockfish.

Two kinds of agreement are measured on a test sample of :func:`~.sidecar.load_sample`:

* static accuracy: Spearman rank correlation and mean absolute error of the predicted expected
  score against the Stockfish expected score (Lichess win-probability scale), and the excess of
  the cross-entropy over the entropy of the label, which is 0 only for a perfect prediction;
* 1-ply best-move agreement: how often the move chosen by the model after one search step equals
  the Stockfish best move.
"""

from collections.abc import Callable

import chess
import numpy as np
import pandas as pd

from chess_engine import _core
from chess_lab.models.target import Representation, white_score


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


def _cross_entropy(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Binary cross-entropy in nats of a predicted score against a soft label, per row.

    With ``pred == truth`` it is the entropy of the label. ``0 * log 0`` counts as 0.

    :param pred: Predicted scores, in ``[0, 1]``.
    :param truth: Labels, in ``[0, 1]``.
    :return: Array of the same length.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        win = np.where(truth > 0, truth * np.log(pred), 0.0)
        loss = np.where(truth < 1, (1 - truth) * np.log(1 - pred), 0.0)
    return -(win + loss)


def static_metrics(pred_white: np.ndarray, true_white: np.ndarray) -> dict:
    """Static accuracy of predicted expected scores.

    The cross-entropy is the training loss on these rows; it cannot go below the entropy of the
    label, so ``excess`` (their difference) is what a better model can still gain.

    :param pred_white: Predicted expected scores for White, in ``[0, 1]``.
    :param true_white: Stockfish expected scores for White, in ``[0, 1]``.
    :return: ``{"spearman", "mae", "cross_entropy", "label_entropy", "excess", "n"}``; mean
        absolute error in expected-score units, the entropies as means in nats; NaN for an
        empty sample.
    """
    pred = np.asarray(pred_white, dtype=np.float64)
    truth = np.asarray(true_white, dtype=np.float64)
    if not pred.size:
        nan = float("nan")
        return {
            "spearman": nan,
            "mae": nan,
            "cross_entropy": nan,
            "label_entropy": nan,
            "excess": nan,
            "n": 0,
        }
    cross = float(np.mean(_cross_entropy(pred, truth)))
    entropy = float(np.mean(_cross_entropy(truth, truth)))
    return {
        "spearman": _spearman(pred, truth),
        "mae": float(np.mean(np.abs(pred - truth))),
        "cross_entropy": cross,
        "label_entropy": entropy,
        "excess": cross - entropy,
        "n": int(pred.size),
    }


#: Score of a mating move: above any evaluation.
MATE_SCORE = 2.0


def move_scores(
    evaluator: _core.Evaluator, board: chess.Board, moves: list[chess.Move]
) -> np.ndarray:
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
            scores.append(MATE_SCORE)
        elif child.is_game_over():
            scores.append(0.5)
        else:
            scores.append(evaluator.evaluate(child.fen(), mover == chess.WHITE))
    return np.array(scores, dtype=np.float64)


def one_ply_choice(evaluator: _core.Evaluator, board: chess.Board) -> chess.Move:
    """Best move of the mover after one ply, judged by the evaluation.

    :param evaluator: Evaluation of the resulting positions, as the engine plays it.
    :param board: Position to move in; it is not modified.
    :return: The legal move with the highest :func:`move_scores` (the first one on a tie).
    :raises ValueError: If the position has no legal move.
    """
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("the position has no legal move")
    return moves[int(np.argmax(move_scores(evaluator, board, moves)))]


def predictions(evaluator: _core.Evaluator, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The evaluation of every position of a table and its Stockfish truth, both White's score.

    :param evaluator: Evaluation to judge, as the engine plays it.
    :param table: Rows with ``fen``, ``cp`` and ``mate``.
    :return: ``(pred, truth)``, one value per row.
    """
    pred = np.array([evaluator.evaluate(fen, True) for fen in table["fen"]], dtype=np.float64)
    truth = white_score(table["cp"].to_numpy(), table["mate"].to_numpy(), Representation())
    return pred, truth


def by_quiet(report: Callable[[np.ndarray], dict], quiet: np.ndarray) -> dict:
    """A report on all rows, on the quiet ones and on the rest.

    :param report: Function of a boolean row mask.
    :param quiet: Boolean array, True on the quiet rows.
    :return: ``{"all": ..., "quiet": ..., "non_quiet": ...}``.
    """
    return {
        "all": report(np.ones(len(quiet), dtype=bool)),
        "quiet": report(quiet),
        "non_quiet": report(~quiet),
    }


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
    pred, truth = predictions(evaluator, test)
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

    return by_quiet(slice_report, quiet)
