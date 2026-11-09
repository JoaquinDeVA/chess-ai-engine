# Incremental network accumulator (`incremental`)

> **What it is:** Updating the first layer of the network move by move instead of recomputing it at every leaf.
>
> **Read it when:** You want to know how a network can be evaluated without summing its first layer again at every leaf.
>
> **Next:** [Table move first](tt-move.md)

## Problem

A network with a sparse first layer spends most of its time in that layer. Its input has one feature per
(piece, square), so the first layer is the sum of one row per piece on the board, and the search summed
all of them again at every leaf although a move changes only two or three pieces.

## Idea

Keep that sum (the accumulator) as part of the position and update it with each move: subtract the rows of the
pieces that leave a square, add the rows of those that arrive. Taking the move back restores the previous sum.

## How it works

The features are relative to the side to move, so two accumulators are kept, one with White as the reference and
one with Black, and the one of the player to move is read. Each move updates both.

**Worked example.** From the starting position, `1. Nf3`: a full evaluation sums 32 rows (one per
piece). The update subtracts the row of "knight on g1" and adds the row of "knight on f3" in each accumulator:
4 rows instead of 32. A capture removes one more row, castling moves two pieces, a promotion swaps a pawn for
another piece; none touches more than six rows.

## Implementation

`NetState` in `rust/src/eval/sparse_net.rs`, behind the `EvalState` trait (`apply`, `apply_null`, `undo`, `evaluate`),
with a stack of snapshots so `undo` is a pop. The search keeps the state beside `GameState` (`Search::play` and
`Search::unplay` in `rust/src/search/node.rs`). Switch: `incremental` (UCI option `Techniques`). Test:
`incremental_equals_full` in `rust/src/eval/sparse_net.rs` (same values as the full evaluation up to rounding).

Only the slotted piece-square networks (`net_pst_*`, the default `net_pst_w32` among them) have an accumulator. The
hybrid networks (`pst_hybrid*` feature sets) sum their whole first layer at each leaf. Until `8700e6f` the
Piece-square-and-terms logistic model (`linear_pst_terms`) had its own incremental state, `LinearState` in
`rust/src/eval/logistic_model.rs`, which kept the four running sums of its table part; it went with the model.

## Measured effect

**`linear_pst_terms`** (2026-11-04, `uci bench` on the six-core laptop, median of 5 runs, same session, commands in
[Rust core](rust-core.md#measured-effect)): 788 k to 928 k nodes per second (+17 %), 275,144 nodes in both.

**SPRT [−5, 0]**, the default set of then (`incremental,tt_move,history,pvs,aspiration,null_move,see`) against the same set without `incremental`. Round robin of 2026-11-06 to 2026-11-09 (fastchess, 10 s + 0.1 s, `Hash` 16, the 10,000 openings of `data/openings/openings.epd`, the model then embedded, `linear_pst_terms`; queue `src/chess_lab/scripts/worker/queue.json` at `d2d03ba`): +35.5 ± 14.7 Elo at 1,690 games, LLR +2.96 of ±2.94, **H1 accepted**: removing it loses (`data/results/roundrobin/abl_incremental/llr.csv`). Measured before `pvs`, `aspiration` and `history` were retired; the set without them has no SPRT of its own yet. The model was `linear_pst_terms`, so the figure is that of
`LinearState`; with `net_pst_w32` the switch was not measured on its own, in games or in `uci bench`.

Rounding can break a tie between two moves, so node counts agree with the full evaluation within 1 %, not exactly.

## Status

Adopted, on by default; since 2026-10-25. Kept in the core on 2026-11-09 (`efb595e`): its removal SPRT accepted H1.
