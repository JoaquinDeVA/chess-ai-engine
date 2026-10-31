# The window

> **What it is:** The window to play against the engine or watch two engines play: what it shows, what every control does and which bot option each setting sets.
>
> **Read it when:** You want to play, or you change the code of `src/chess_engine/gui/`.
>
> **Next:** [Results](RESULTS.md)

## Problem

The engine had only a console game with a long list of options, and moves typed in UCI. A window
lets a person play on a board and choose who plays whom, and keeps the options of the bots on one
screen instead of on the command line. Games between engines run without the window, with fastchess.

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

Every setting is a field of `AISettings` (`gui/settings.py`). The engine is the UCI binary
`rust/target/release/uci`, which has one evaluation (`linear_pst_terms`, compiled in) and a fixed
quiescence depth of 4 plies, so neither is a setting.

| Setting | `AISettings` field | UCI option or `go` limit | Default |
|---|---|---|---|
| Maximum depth (0 = no limit: the time decides) | `max_depth` | `go depth N` | no limit |
| Fixed time per move; unchecked, the engine spends a share of its clock | `move_time` | `go movetime ms` | 3 s |
| Opening book | `opening_book` | none: read in Python before the engine | on |
| Endgame tablebases (up to five pieces) | `tablebase` | none: read in Python before the engine | on |
| Advanced: search techniques | `techniques` | `setoption name Techniques` | all but `lmr` |
| Advanced: search threads | `threads` | `setoption name Threads` | up to 4 |

With a clock and no fixed time, the window sends `wtime`, `btime`, `winc` and `binc`; the binary
does not use the increment yet (see [Architecture](ARCHITECTURE.md)).

**Implementation.** The bot searches on a `QThread` (`game_window.py: MoveThread`), so the window
does not freeze: the UCI binary is a separate process, and `UciBot` (`chess_engine/uci.py`) waits for its
`bestmove`. Each game builds new bots; an answer to an old request (after an undo or a resignation) is ignored.
`game_model.py` (position, SAN list, clocks, result, undo) and `settings.py` do not import Qt and are
tested without a window (`tests/app/gui/test_gui_model.py`); the widgets are tested offscreen
(`tests/app/gui/test_gui_widgets.py`, skipped if PySide6 is missing).

**Measured effect.** Not measured: the window does not change the engine.

**Status.** Implemented. Not included: an evaluation bar, human against human, saving and loading games.

## Pieces and licence

The pieces are the *cburnett* set from the Lichess repository, in `src/chess_engine/gui/assets/pieces/`
with their licence (GPLv2+, Colin M.L. Burnett). The colours and the layout imitate chess.com; none of
its images are used. The code of the project is MIT.
