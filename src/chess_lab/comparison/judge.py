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

from chess_engine.models.base import EvaluationModel
from chess_engine.models.representation import Representation, white_score

#: Positions per call of ``predict_cp_batch`` when judging a whole sample.
CHUNK_SIZE = 4096


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


def one_ply_choice(model: EvaluationModel, board: chess.Board) -> chess.Move:
    """Best move of the mover after one ply, judged by the model.

    All the non-terminal children are evaluated with a single ``predict_cp_batch`` call. A
    mating move ranks first; stalemate and other draws count as 0 centipawns.

    :param model: Model that evaluates the resulting positions.
    :param board: Position to move in; it is not modified.
    :return: The legal move with the highest evaluation for the side to move.
    :raises ValueError: If the position has no legal move.
    """
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("the position has no legal move")
    sign = 1.0 if board.turn == chess.WHITE else -1.0
    scores = [0.0] * len(moves)
    children, indices = [], []
    for index, move in enumerate(moves):
        child = board.copy(stack=False)
        child.push(move)
        if child.is_checkmate():
            scores[index] = float("inf")
        elif child.is_game_over():
            scores[index] = 0.0
        else:
            children.append(child)
            indices.append(index)
    if children:
        for index, cp in zip(indices, model.predict_cp_batch(children), strict=True):
            scores[index] = sign * float(cp)
    return moves[max(range(len(moves)), key=lambda i: scores[i])]


def _predict_white(model: EvaluationModel, fens: list[str]) -> np.ndarray:
    """Predicted expected scores for White, computed in chunks of :data:`CHUNK_SIZE`."""
    rep = Representation()
    parts = []
    for start in range(0, len(fens), CHUNK_SIZE):
        boards = [chess.Board(fen) for fen in fens[start : start + CHUNK_SIZE]]
        parts.append(white_score(model.predict_cp_batch(boards), np.nan, rep))
    return np.concatenate(parts) if parts else np.empty(0)


def judge(model: EvaluationModel, test: pd.DataFrame, agreement_n: int = 20_000) -> dict:
    """Judge a model on a test sample, on all positions and on each quiet slice.

    Static metrics use every row; the Stockfish truth is always on the Lichess win-probability
    scale. Best-move agreement uses the first ``agreement_n`` rows of the sample (rows without a
    best move are skipped) and is split by slice with the same rows.

    :param model: Model to judge.
    :param test: Sample with columns ``fen``, ``cp``, ``mate``, ``best_move``, ``quiet``.
    :param agreement_n: Number of leading rows used for best-move agreement.
    :return: ``{"all": ..., "quiet": ..., "non_quiet": ...}``, each with the keys of
        :func:`static_metrics` plus ``best_move_agreement`` (NaN if no row) and
        ``agreement_n`` (rows with a best move that were compared).
    """
    fens = test["fen"].tolist()
    pred = _predict_white(model, fens)
    truth = white_score(test["cp"].to_numpy(), test["mate"].to_numpy(), Representation())
    quiet = test["quiet"].to_numpy(dtype=bool)

    head = min(agreement_n, len(fens))
    agree = np.empty(head, dtype=np.float64)
    for row in range(head):
        best = test["best_move"].iloc[row]
        if best is None or (isinstance(best, float) and np.isnan(best)):
            agree[row] = np.nan
            continue
        choice = one_ply_choice(model, chess.Board(fens[row]))
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
