# Make/unmake and fast terminal detection

> **What it is:** Playing moves in place on one state, with cheap position keys and a cheap answer to "is the game over?".
>
> **Read it when:** You want to know why the board costs little per node.
>
> **Next:** [Opening book](opening-book.md)

### Make/unmake on one state, with fast transposition keys

**Problem.** The first version copied the whole game state at every node and built the position key
from a FEN string, which costs time at every node.

**What it does.** The search plays a move in place on one `GameState`, searches the child and
takes the move back; the state keeps an undo stack (position, history). The transposition key is
a hash of the position plus the two move counters, which identifies the same positions as the FEN
without building the string.

**How it is implemented.** `GameState::apply`, `undo` and `key` in `rust/src/board.rs`; the
`apply` / `undo` pairs in `Search::node` and `Search::quiescence`.

**Measured effect.** Commit `ea86b20` reports, on the fixed benchmark at depth 3 of the Python
engine, identical moves and node counts and a total time from 10.8 s to 3.7 s (2.9 times faster),
*together with* the change of the terminal-detection section.

**Status.** Kept; since 2026-10-16 (commit `ea86b20`), now in Rust.

### Fast terminal detection with the same draw rules

**Problem.** "Is the game over here?" is asked at every node, and the reference rule,
`board.outcome(claim_draw=True)` of python-chess, replays the game to claim a threefold repetition.

**What it does.** Same answer as `board.outcome(claim_draw=True)`, but the repetition is read from the
keys of the positions since the last irreversible move, which the state extends in constant time per
move. Checkmate, stalemate and insufficient material come from the rules of the position; the
fifty-move claim needs 100 halfmoves and a legal move; the threefold claim holds when the position
occurred three times or a legal move reaches one that occurred twice.

**How it is implemented.** `GameState::outcome` and `can_claim_threefold` in `rust/src/board.rs`;
the history is rebuilt on every call from the root FEN and the moves played. Tests:
`rust/tests/rules.rs` (end of game by mate, stalemate, insufficient material, fifty moves and repetition).

**Measured effect.** Commit `ba3d5cb` reports, at depth 3 on the fixed benchmark of the Python
engine, the same moves and nodes as the baseline and a search 1.89 times faster (the 10.8 s to 3.7 s
of the make/unmake section is the combined effect of both changes).

**Status.** Kept; since 2026-10-16 (commit `ba3d5cb`), now in Rust.
