# Late move pruning (`lmp`)

> **What it is:** At a shallow depth, not searching the quiet moves beyond a count that grows with the depth.
>
> **Read it when:** You want to know the pruning that saves most nodes, and why it is not on by default.
>
> **Next:** [Search techniques](README.md)

## Problem

At 1 to 3 plies from the horizon the moves at the end of the order almost never turn out to be the best one,
yet each costs a search.

## Idea

Trust the order even more than [late move reductions](lmr.md) do: past a number of quiet moves, skip them
instead of searching them shallower.

## How it works

At most 3 plies left, not in check, `alpha > 0.01` and `beta < 0.99`: the quiet moves (not a capture, promotion, killer or
check) from the `3 + depth²`-th on are skipped (index counted from 0). That is 4 moves at 1 ply left, 7 at 2 and 12 at 3.

**Worked example.** A node with 2 plies left and 30 legal moves, 22 of them quiet. The first 7 moves of the order are
searched (captures and killers come first); of the remaining quiet ones, those that do not give check are skipped, instead of 15 searches.

## Implementation

The `lmp` check in the move loop of `Search::node`, `rust/src/search/node.rs` (`late_move_limit`). Switch `lmp`. Tests in
`rust/src/search/tests.rs`: the limits per depth, and the forced mate and the hanging queen unchanged.

## Measured effect

`TTD_TECHNIQUES=<set> ./target/release/uci ttd 11 bench/middlegame.epd 60` from `rust/`, 1 thread, `net_pst_w32`, 2026-11-10, the set
`incremental,tt_move,null_move,see,lmr,rfp`: 215,120,644 nodes and a median of 8.65 s without it; 66,027,302 nodes (−69.3 %) and 3.91 s with it. With `futility` as well:
66,938,464 nodes and 4.55 s: together they are no better than `lmp` alone. The branching factor rises from 2.66 to 2.75 (2.95 with both). So big a fall in nodes
may also be dropping good moves; only games can say. Elo effect: not measured, SPRT queued for the node.

## Status

Implemented, **off by default** pending its SPRT [−5, 0] on the node (2026-11-10).
