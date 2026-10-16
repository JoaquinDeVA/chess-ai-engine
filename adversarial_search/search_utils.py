"""Reusable building blocks for the advanced alpha-beta search.

The helpers in this module are deliberately independent of chess so that they can be unit tested
with hand-made game trees:

* a **transposition table** (``dict`` of :class:`TTEntry`) with bound flags and a size cap,
* **killer moves** (quiet moves that recently caused a beta cut-off at a given depth),
* **MVV-LVA** capture ordering (Most Valuable Victim, Least Valuable Attacker).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Hashable, List, Optional, Tuple

import chess

if TYPE_CHECKING:  # pragma: no cover - import only needed for type checkers
    from .game_algorithms import AdversarialSearchResult

#: The stored value is the exact minimax value of the position.
EXACT = "EXACT"
#: The stored value is a lower bound (the search failed high / beta cut-off).
LOWER_BOUND = "LOWERBOUND"
#: The stored value is an upper bound (the search failed low).
UPPER_BOUND = "UPPERBOUND"

#: Default maximum number of entries kept in a transposition table.
DEFAULT_MAX_TT_ENTRIES = 500_000
#: Number of killer moves remembered per depth.
MAX_KILLERS_PER_DEPTH = 3

_PIECE_ORDER = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}


@dataclass
class TTEntry:
    """Entry of a transposition table.

    :param result: Search result (value and best move) stored for the position.
    :param depth_remaining: Remaining search depth when the entry was computed.
    :param flag: One of :data:`EXACT`, :data:`LOWER_BOUND` or :data:`UPPER_BOUND`.
    """

    result: "AdversarialSearchResult"
    depth_remaining: int
    flag: str


def probe_transposition_table(
    table: Dict[Hashable, TTEntry],
    key: Hashable,
    depth_remaining: int,
    alpha: float,
    beta: float,
) -> Tuple[Optional["AdversarialSearchResult"], float, float]:
    """Look a position up in the transposition table.

    :param table: Transposition table.
    :param key: Hashable key of the position.
    :param depth_remaining: Remaining depth of the search that is probing.
    :param alpha: Current alpha bound.
    :param beta: Current beta bound.
    :return: ``(result, alpha, beta)``. ``result`` is not ``None`` when the entry allows the
        search to stop at this node; otherwise the (possibly tightened) window is returned.
    """
    entry = table.get(key)
    if entry is None or entry.depth_remaining < depth_remaining:
        return None, alpha, beta

    if entry.flag == EXACT:
        return entry.result, alpha, beta
    if entry.flag == LOWER_BOUND:
        alpha = max(alpha, entry.result.value)
    elif entry.flag == UPPER_BOUND:
        beta = min(beta, entry.result.value)

    if alpha >= beta:
        return entry.result, alpha, beta
    return None, alpha, beta


def store_transposition_entry(
    table: Dict[Hashable, TTEntry],
    key: Hashable,
    result: "AdversarialSearchResult",
    depth_remaining: int,
    alpha_orig: float,
    beta_orig: float,
    max_entries: int = DEFAULT_MAX_TT_ENTRIES,
) -> None:
    """Store a search result, classifying it against the *original* alpha-beta window.

    When the table is full the oldest entries are dropped (dicts keep insertion order), which
    keeps memory bounded during long games.

    :param table: Transposition table (modified in place).
    :param key: Hashable key of the position.
    :param result: Value and best move found for the position.
    :param depth_remaining: Remaining depth the value was computed with.
    :param alpha_orig: Alpha bound on entry to the node, before the search updated it.
    :param beta_orig: Beta bound on entry to the node.
    :param max_entries: Maximum table size.
    """
    if result.value <= alpha_orig:
        flag = UPPER_BOUND
    elif result.value >= beta_orig:
        flag = LOWER_BOUND
    else:
        flag = EXACT

    table.pop(key, None)  # re-insert so that fresh entries are the last to be evicted
    table[key] = TTEntry(result=result, depth_remaining=depth_remaining, flag=flag)

    overflow = len(table) - max_entries
    if overflow > 0:
        for old_key in list(table)[:overflow]:
            del table[old_key]


def record_killer_move(killers: Dict[int, List[Any]], depth: int, move: Any) -> None:
    """Remember ``move`` as a killer move for ``depth`` (at most three per depth).

    :param killers: Mapping ``depth -> list of moves`` (modified in place).
    :param depth: Depth the cut-off happened at.
    :param move: Move that produced the cut-off.
    """
    moves = killers.setdefault(depth, [])
    if move in moves:
        return
    moves.append(move)
    if len(moves) > MAX_KILLERS_PER_DEPTH:
        moves.pop(0)


def reorder_by_killer_moves(moves: List[Any], depth: int, killers: Dict[int, List[Any]]) -> List[Any]:
    """Move the killer moves of ``depth`` to the front, keeping the rest in order.

    :param moves: Candidate moves.
    :param depth: Depth the killers were recorded at.
    :param killers: Mapping ``depth -> killer moves``.
    :return: The reordered list (a new list; ``moves`` is not modified).
    """
    depth_killers = killers.get(depth, [])
    return sorted(moves, key=lambda move: move not in depth_killers)


def mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    """Score a move with Most-Valuable-Victim / Least-Valuable-Attacker.

    En-passant captures are scored as pawn captures (the victim is not on the target square).

    :param board: Position the move is played from.
    :param move: Move to score.
    :return: ``0`` for quiet moves, otherwise a higher number for more attractive captures.
    """
    attacker = board.piece_at(move.from_square)
    if attacker is None:
        return 0

    if board.is_en_passant(move):
        victim_type = chess.PAWN
    else:
        victim = board.piece_at(move.to_square)
        if victim is None:
            return 0
        victim_type = victim.piece_type

    return 10 * _PIECE_ORDER[victim_type] - _PIECE_ORDER[attacker.piece_type]


def order_by_mvv_lva(board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
    """Order moves with captures first (best MVV-LVA first), quiet moves keep their order.

    Keeping the incoming order of the quiet moves preserves any previous ordering, such as
    killer moves placed in front.

    :param board: Position the moves are played from.
    :param moves: Legal moves.
    :return: Reordered list.
    """
    captures = [move for move in moves if board.is_capture(move)]
    quiets = [move for move in moves if not board.is_capture(move)]
    captures.sort(key=lambda move: mvv_lva_score(board, move), reverse=True)
    return captures + quiets
