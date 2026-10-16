# Adding a bot

A bot is one file in `bots/`. Subclass one of the bases in `bots/_search_bots.py` and implement
`evaluate(state, player)`, the expected score of `player` in `state`, strictly between 0 and 1.

```python
# bots/material_bot.py
import chess

from bots._search_bots import IterativeDeepeningBot
from evaluation.features import extract_material_features


class MaterialBot(IterativeDeepeningBot):
    """Iterative-deepening alpha-beta that only counts material."""

    def __init__(self, max_depth: int = 2):
        super().__init__("MaterialBot", max_depth=max_depth)

    def evaluate(self, state, player):
        white_score = extract_material_features(state.get_board())["material_diff_normalized"]
        score = white_score if player == chess.WHITE else 1.0 - white_score
        return min(0.99, max(0.01, score))
```

That is all: the registry finds it, so it appears in `--list-bots` and in tournaments.

```bash
python main.py --list-bots
python main.py --player1 MaterialBot --player2 IterativeHeuristicBot --depth 3
```

## Choosing a base

| Base | Use it when | You get |
|---|---|---|
| `IterativeDeepeningBot` | you want the strongest play for the time available | iterative deepening, transposition table, killer moves, MVV-LVA ordering, time budget |
| `ChessBot` | you do not want a tree search at all | just `get_move(board, time_limit)` |

## Rules of the registry

* Every public `*.py` in `bots/` is imported; files starting with `_` are not (use that for shared code).
* Every `ChessBot` subclass defined in a file is registered under its class name.
* Constructors must work without arguments; extra keyword arguments (`max_depth`, `model`) can be passed
  with `registry.get_bot(name, **options)` and the `--depth` / `--model` CLI options.
* Override `order_moves(board, moves)` to give the search a better move order, `on_game_start` to reset
  per-game state, and `get_info()` to describe the bot.

## Test it

Copy the pattern of `tests/test_bots.py`: legal moves from several positions, mate in one
(`tests.conftest.MATE_IN_ONE`) and the board is left untouched.
