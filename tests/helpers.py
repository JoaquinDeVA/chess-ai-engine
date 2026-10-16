"""Tiny hand-built game trees used to test the search algorithms deterministically."""

from typing import Any, Dict, Hashable, List, Optional

from chess_engine.search.game_problems import GameProblem, GameState


class TreeState(GameState):
    """Game state that is a node of an explicit tree.

    :param name: Unique node name.
    :param tree: Mapping ``node -> {action: child}``; leaves are absent or empty.
    :param utilities: Mapping ``leaf -> utility for player 0`` (1 win, 0.5 draw, 0 loss).
    :param to_move: Player to move at this node (0 or 1).
    """

    def __init__(
        self,
        name: str,
        tree: Dict[str, Dict[str, str]],
        utilities: Dict[str, float],
        to_move: int = 0,
    ):
        self.name = name
        self.tree = tree
        self.utilities = utilities
        self.to_move = to_move
        self._undo_stack: List[Any] = []

    def get_legal_moves(self) -> List[Any]:
        return list(self.tree.get(self.name, {}).keys())

    def make_move(self, move: Any) -> "TreeState":
        self.name = self.tree[self.name][move]
        self.to_move = 1 - self.to_move
        return self

    def apply_move(self, move: Any) -> None:
        self._undo_stack.append((self.name, self.to_move))
        self.make_move(move)

    def undo_move(self) -> None:
        self.name, self.to_move = self._undo_stack.pop()

    def is_over(self) -> bool:
        return not self.tree.get(self.name)

    def get_current_player(self) -> int:
        return self.to_move

    def get_winner(self) -> Optional[Any]:
        return None

    def copy(self) -> "TreeState":
        return TreeState(self.name, self.tree, self.utilities, self.to_move)

    def get_board(self) -> None:
        return None

    def __str__(self) -> str:
        return self.name

    def key(self) -> Hashable:
        return (self.name, self.to_move)


class TreeGame(GameProblem):
    """Game over a :class:`TreeState`; utilities are given for player 0 and flipped for player 1."""

    def utility(self, state: TreeState, player: Any) -> float:
        value = state.utilities[state.name]
        return value if player == 0 else 1.0 - value


def never_cutoff(state: Any, depth: int, elapsed: float) -> bool:
    """Cutoff test that never stops the search early."""
    return False


def zero_eval(state: Any, player: Any) -> float:
    """Neutral evaluation used when only terminal values matter."""
    return 0.5
