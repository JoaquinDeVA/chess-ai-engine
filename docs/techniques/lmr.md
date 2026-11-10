# Late move reductions (`lmr`)

> **What it is:** Searching the quiet moves far down the move order less deeply first, and again at full depth only if they turn out to improve on the best move.
>
> **Read it when:** You want to know why the search reaches depth 11 in half the time, and what it risks.
>
> **Next:** [Reverse futility pruning](rfp.md)

## Problem

With a good move order the best move is almost always among the first few. The moves tried last are usually
worse, yet each costs a full-depth search to prove it.

## Idea

Trust the order: search the late quiet moves shallower. If one of them proves better than expected (it beats the
best value so far even at reduced depth) it was a mistake to reduce it, so search it again at full depth.

## How it works

A move is reduced when it is the fourth move of the node or later (`index ≥ 3`), at least 3 plies are left, and
it is quiet: not a capture, not a promotion, not a killer, does not give check, and the node is not in check.
The reduction is `0.75 + ln(depth) · ln(index) / 2.25`, truncated, one ply less when the history score of the
move is above zero (it has cut before), and never so large that less than one ply is left.

**Worked example.** A node with 6 plies left, the move at index 10: `0.75 + ln 6 · ln 10 / 2.25 = 0.75 + 1.834 = 2.58`,
so 2 plies. The normal child search has 5 plies; this one has `5 − 2 = 3`. If it still fails to beat the best
value, it is done: a 3-ply search instead of a 5-ply one. If it beats it, the move is searched again at 5 plies
and, as always, with the full window if it is better than the best.

## Implementation

`reduction` and the late-move branch of `Search::child_value` in `rust/src/search/node.rs`
(`LMR_FROM_DEPTH = 3`, `LMR_FROM_INDEX = 3`); the caller in `Search::node` decides which moves are reducible.
Switch `lmr`. Tests in `rust/src/search/tests.rs`: the reduction table and a forced mate still found.
The «good history» threshold (score above zero) is a choice of this implementation, not measured.

## Measured effect

`TTD_TECHNIQUES=<set> ./target/release/uci ttd 11 bench/middlegame.epd 120` from `rust/`, 1 thread, `net_pst_w32`,
20 positions, desktop, 2026-11-10: without `lmr` 1,004,922,392 nodes and a median of 23.53 s; with it 324,595,891
nodes (−68 %) and 10.82 s, branching factor 2.83 → 2.75.

Local screening, the same binary with and without the switch (300 games, 10 s + 0.1 s, `Hash` 16, `-srand 12`):
51.33 % (+134 −126 =40), +9 ± 36 Elo (estimated; it only sees large losses). No SPRT yet: queued for the node.

An earlier `lmr`, removed in the Python era, lost −8.6 ± 6.1 Elo at 20,000 nodes per move
(`data/results/matches/lmr_nodes/match.log`): a budget of nodes and no `history`, a different regime from this one.

## Status

On by default since 2026-11-10, pending its SPRT [−5, 0] on the node.
