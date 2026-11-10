# Internal iterative reduction (`iir`)

> **What it is:** Searching a node one ply shallower when the table has no move for it.
>
> **Read it when:** You want to know why the iterative deepening got cheaper without losing the order of the moves.
>
> **Next:** [Futility pruning](futility.md)

## Problem

A node with no table move is ordered only by captures, killers and history. That order is the worst the search has, and the
node is searched at full depth with it.

## Idea

Spend one ply less on such a node. On the next iteration the table holds the best move of this search, the order is
good and the node is searched at its full depth.

## How it works

At 4 plies left or more, not at the root, with no best move in the table entry (`table_move` empty), the depth of the node
drops by one before its moves are searched.

**Worked example.** A node with 6 plies left and no table move is searched with 5. The best move found is stored; the next
iteration reaches the same node with 7 plies left, finds the move first and searches it with the full 7.

## Implementation

The `depth_remaining` shadowing before the move loop of `Search::node`, `rust/src/search/node.rs` (`IIR_FROM_DEPTH`). Switch `iir`.
Test: `internal_iterative_reduction_keeps_the_forced_mate_and_the_hanging_queen` in `rust/src/search/tests.rs`.

## Measured effect

`TTD_TECHNIQUES=<set> ./target/release/uci ttd 11 bench/middlegame.epd 60` from `rust/`, 1 thread, `net_pst_w32`, 2026-11-10, set
`incremental,tt_move,null_move,see,lmr,rfp`: 215,120,644 nodes and 8.22 s median without it; 108,502,704 nodes (-49.6 %) and 4.15 s with it.
At depth 13 (`ttd 13 bench/middlegame.epd 60`): 64.90 s median (14 of 20 finished within 120 s) without it, 29.73 s (19 of 20) with it.
Elo effect: not measured, SPRT queued for the node.

## Status

Implemented, **on by default** pending its SPRT [-5, 0] on the node (2026-11-10).
