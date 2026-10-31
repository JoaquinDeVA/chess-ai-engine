# Make/unmake and fast terminal detection

> **What it is:** Playing moves in place on one state, with cheap position keys and a cheap answer to "is the game over?".
>
> **Read it when:** You want to know why the board costs little per node.
>
> **Next:** [Opening book](opening-book.md)

## Problem

Two questions are asked at every node: "what does the position look like after this move?" and "is the game
over here?". The first version copied the whole game state for every child, built the position key from a FEN
string, and answered the second by replaying the game to claim a threefold repetition
(`board.outcome(claim_draw=True)` of python-chess). All three cost time at every node.

## Idea

Keep one state and change it in place: play the move, search the child, take the move back. Keep, beside the
state, the keys of the positions since the last irreversible move (capture or pawn move); a repetition can only
happen among those, so the draw rules read that short list instead of replaying the game.

## How it works

- **Make/unmake.** `apply` pushes the move onto an undo stack and updates the position; `undo` pops it.
- **Key.** A hash of the position plus the two move counters: the same positions the FEN identifies, without
  building the string.
- **Terminal check**, same answer as `board.outcome(claim_draw=True)`: checkmate, stalemate and insufficient
  material come from the position; the fifty-move claim needs 100 halfmoves and a legal move; the threefold
  claim holds when the position occurred three times, or a legal move reaches one that occurred twice.

**Worked example.** After 1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 the list of keys since the last irreversible
move holds 8 positions, in which the position after Black's `Nf6` appears twice. Black to move asks: does a legal
move reach a position seen twice? `Ng8` reaches the start position, seen twice (move 0 and after 2... Ng8): yes, a
threefold claim is available. The check reads eight keys, not the whole game.

## Implementation

`GameState::apply`, `undo`, `key`, `outcome` and `can_claim_threefold` in `rust/src/board.rs`; the `apply`/`undo`
pairs in `Search::node` and `Search::quiescence` (`rust/src/search/node.rs`). Tests: `rust/tests/rules.rs` (mate,
stalemate, insufficient material, fifty moves and repetition). Always on.

## Measured effect

Measured with the earlier Python engine, on its fixed benchmark at depth 3: the two changes together took the
total time from 10.8 s to 3.7 s (2.9 times faster), with identical moves and node counts (commit messages of
`ea86b20` and `ba3d5cb`; `git show ea86b20`). The terminal detection alone made the search 1.89 times faster.

## Status

Adopted; since 2026-10-16 (`ea86b20`, `ba3d5cb`), in Rust since 2026-10-25.
