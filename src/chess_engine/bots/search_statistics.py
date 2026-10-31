"""What a search measured."""

from typing import Any, Dict, NamedTuple, Tuple


class SearchStatistics(NamedTuple):
    """
    Statistics collected during a search.

    :param nodes_visited: Total number of nodes evaluated
    :type nodes_visited: int
    :param pruning_count: Number of times alpha-beta pruning occurred
    :type pruning_count: int
    :param max_depth_reached: Maximum depth reached in search
    :type max_depth_reached: int
    :param truncated: True if the search was stopped early by the clock so its result may be based
        on a shallower search than asked
    :type truncated: bool
    :param completed_depth: Depth of the deepest iterative-deepening iteration that finished
        without being truncated (0 if none did); ``max_depth_reached`` also counts plies seen
        in a truncated iteration
    :type completed_depth: int
    :param iterations: One ``(depth, seconds, nodes, truncated)`` entry per iteration that was
        started, oldest first
    :type iterations: Tuple[Tuple[int, float, int, bool], ...]
    :param stop_reason: Why the deepening ended: ``"max_depth"``, ``"estimate"`` (the next
        iteration would not fit), ``"budget"`` (the soft budget is used up), ``"truncated"``
        (the hard limit interrupted an iteration), ``"book"`` or ``"tablebase"`` (no search)
    :type stop_reason: str
    :param stop_predicted: Seconds predicted for the iteration that the estimate refused to start
        (0 unless ``stop_reason`` is ``"estimate"``)
    :type stop_predicted: float
    """

    nodes_visited: int
    pruning_count: int
    max_depth_reached: int
    truncated: bool = False
    completed_depth: int = 0
    iterations: Tuple[Tuple[int, float, int, bool], ...] = ()
    stop_reason: str = ""
    stop_predicted: float = 0.0

    @classmethod
    def from_core(cls, raw: Dict[str, Any]) -> "SearchStatistics":
        """Statistics from the dictionary the Rust core returns.

        :param raw: Output of ``Searcher.search``.
        :return: The statistics.
        """
        return cls(
            raw["nodes_visited"],
            raw["pruning_count"],
            raw["max_depth_reached"],
            raw["truncated"],
            raw["completed_depth"],
            tuple(tuple(item) for item in raw["iterations"]),
            raw["stop_reason"],
            raw["stop_predicted"],
        )
