# The window

> **What it is:** The window to play against the engine or watch two engines play: what it shows, what every control does and which bot option each setting sets.
>
> **Read it when:** You want to play, or you change the code of `src/chess_engine/gui/`.
>
> **Next:** [Results](RESULTS.md)

## Problem

The engine had only a console game with a long list of options, and moves typed in UCI. A window
lets a person play on a board and choose who plays whom, and keeps the options of the bots on one
screen instead of on the command line. Tournaments between bots, which need no window, have their
own entry point.

## What it does

`python main.py` opens the window (it needs the optional dependency: `pip install -e ".[gui]"`,
which installs PySide6). It has two pages.

**Player selection.** For White and for Black you choose *Human* or *AI*; the **Settings** button of an
AI opens its settings. You also choose the time control (no clock, or one of the presets from one
to thirty minutes, with or without increment). *Swap sides* exchanges the two players with their
settings. Two humans are not supported.

**Game.** The board shows the coordinates inside the squares, the last move, the legal moves of the
selected piece (a dot, or a ring on a capture) and a red glow on a king in check. A piece moves by
clicking it and then its target, or by dragging it; a pawn that reaches the last rank asks which
piece it becomes. The side panel has the move list in SAN (click a move, or use the arrow keys,
Home and End, to look at an earlier position; the game goes on in the live position), and the
buttons:

| Button | Effect |
|---|---|
| Undo | Takes back your last move and the answer of the AI (both clocks go back too); only on your turn |
| Draw | Claims the draw if the position allows it (repetition, fifty moves); otherwise offers it, and the AI accepts when it has no more material than you |
| Resign | You lose |
| Flip | Turns the board |
| Copy PGN | Puts the game, in PGN, on the clipboard |
| Rematch / New game | Same players again / back to the player selection (shown when the game is over) |

When two AIs play, a speed menu (Fast, Normal, Slow) sets the pause between moves; the clocks
stop during the pause. A clock that reaches zero loses the game, unless the other side cannot
mate, which is a draw.

## AI settings

Every setting is an option of `SearchBot(**options)` (`gui/settings.py`:
`AISettings.bot_options`). The defaults are those of the bots.

| Setting | Bot option | Default |
|---|---|---|
| Evaluation: the hand-written heuristic `h1` or a model whose file exists, shown as full name and id, e.g. "Piece-square-and-terms logistic model (`linear_pst_terms`)" | `evaluation` | `linear_pst_terms` |
| Maximum depth (0 = no limit: the time decides) | `max_depth` | no limit |
| Fixed time per move; unchecked, the bot spends a share of its clock | `time_manager` | 3 s |
| Opening book | `opening_book` | on |
| Endgame tablebases (up to five pieces) | `tablebase` | on |
| Advanced: quiescence plies | `quiescence_plies` | 4 |
| Advanced: search techniques | `techniques` | all but `lmr` |
| Advanced: search threads | `threads` | up to 4 |

**Implementation.** The bot searches on a `QThread` (`game_window.py: MoveThread`), so the window
does not freeze: the Rust search releases the Python lock while it runs. Each game builds new
bots; an answer to an old request (after an undo or a resignation) is ignored.
`game_model.py` (position, SAN list, clocks, result, undo) and `settings.py` do not import Qt and are
tested without a window (`tests/app/gui/test_gui_model.py`); the widgets are tested offscreen
(`tests/app/gui/test_gui_widgets.py`, skipped if PySide6 is missing).

**Measured effect.** Not measured: the window does not change the engine.

**Status.** Implemented. Not included: an evaluation bar, human against human, saving and loading games.

## Pieces and licence

The pieces are the *cburnett* set from the Lichess repository, in `src/chess_engine/gui/assets/pieces/`
with their licence (GPLv2+, Colin M.L. Burnett). The colours and the layout imitate chess.com; none of
its images are used. The code of the project is MIT.
