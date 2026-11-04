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
`incremental_equals_full` in `rust/src/eval/sparse_net.rs` (same values as the full evaluation up to rounding). It does
nothing for the logistic models, which are already cheap sums.

## Measured effect

Not measured with the light network candidates ([Models](../MODELS.md#light-network-candidates-net_pst_w32-net_pst_w64-net_pst_w64x16)), which are not trained yet. Rounding can break a tie between two moves, so node counts agree with the full evaluation within 1 %, not exactly.

## Status

Adopted, on by default; since 2026-10-25.
