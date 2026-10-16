# Tactical patterns

`TacticalAlphaBetaBot` asks a Prolog knowledge base, `knowledge/chess_patterns.pl`, which legal
moves *create* a tactical pattern, and searches those first. Better ordering means earlier
alpha-beta cut-offs: the bot returns the same move as `NonTacticalAlphaBetaBot` while visiting
fewer nodes.

The Python side (`PrologPatternDetector`) loads the position once as facts (`piece/4`,
`to_move/1`, `can_castle/2`, `en_passant/2`, `move/4`), queries each predicate once for all moves
and scores the moves. Coordinates are 1-based `(row, column)`.

| Pattern | Predicate | Idea | Ordering score |
|---|---|---|---|
| Absolute pin | `move_creates_absolute_pin/5` | the pinned piece cannot move: it shields the king | `50 + 3·score` |
| Relative pin | `move_creates_relative_pin/5` | moving the pinned piece loses something more valuable | `30 + 3·score` |
| Fork | `move_creates_fork/5` | one piece attacks two or more valuable enemy pieces | `100 + 3·score` |
| Skewer | `move_creates_skewer/5` | a valuable piece must move and exposes a weaker one behind it | `80 + 3·score` |
| Capture | `move_creates_capture/5` | most valuable victim, least valuable attacker | `3·score` |
| Promotion | `move_creates_promotion/4` | pawn reaches the last rank | `+800` |
| Check | (python-chess) | includes discovered checks | `+50` |

`score` is computed in Prolog from the piece values, so patterns involving a king or a queen rank
above those involving minor pieces.

| | |
|---|---|
| ![Absolute pin](img/absolute_pin.svg) | ![Relative pin](img/relative_pin.svg) |
| **Absolute pin**: the rook pins the knight to the king. | **Relative pin**: the knight is pinned to the queen behind it. |
| ![Fork](img/fork.svg) | ![Skewer](img/skewer.svg) |
| **Fork**: the knight attacks the king and the rook. | **Skewer**: the rook checks the king, which must leave the rank, exposing the rook behind it. |

The diagrams are generated with `python scripts/render_pattern_boards.py`.

## Requirements

SWI-Prolog (`apt install swi-prolog`) and `pip install -e ".[prolog]"`. Without them the bot is
listed but cannot be loaded, and the Prolog tests are skipped.
