# Killer moves

> **What it is:** Remembering the moves that caused a cut-off at a depth and trying them first in sibling nodes.
>
> **Read it when:** You want to know how quiet moves are ordered.
>
> **Next:** [Move ordering by MVV-LVA](mvv-lva.md)

## Problem

Alpha-beta skips most when the best move is tried first. For captures there is a cheap guess
([MVV-LVA](mvv-lva.md)); for quiet moves there is none, so a refuting quiet move may be tried last.

## Idea

A quiet move that refutes one line often refutes its sibling lines too: if `Ng5` punishes one of the
opponent's moves at a depth, it probably punishes the next one at the same depth. Remember it and try it
early.

## How it works

When a move causes a cut-off (`alpha ≥ beta`), it is stored as a killer for the remaining depth; up to three
are kept per depth, the oldest dropped first. In later nodes at that depth the killers that are legal there
go right after the captures and before the other quiet moves.

**Worked example.** At depth 3 the opponent has three moves, A, B and C. Under A, White's quiet move
`Ng5` reaches `beta` as the fourth move tried: cut-off, `Ng5` becomes a killer for depth 2. Under B the
order is captures, then `Ng5`, then the other quiet moves; if `Ng5` refutes B as well, the node is cut after
one quiet move instead of four. Under C, where `Ng5` is not legal, the killer is skipped and nothing is lost.

## Implementation

`Killers::record` and the killer step of `order` in `rust/src/search/ordering.rs`, called from `Search::node`
(`rust/src/search/node.rs`). `MAX_KILLERS_PER_DEPTH = 3`. A killer is identified by its squares and promotion,
not by what it captures. The killers are shared across the iterations and moves of a game and cleared on a new
game. Always on (not a switch).

## Measured effect

Not measured with the Rust engine as a whole. Storing the killers by ply instead of by remaining depth (tried 2026-11-10 on `uci ttd 9 rust/bench/middlegame.epd`, one thread) raised the nodes from 107,372,423 to 111,175,719 (+3.5 %) with the same killer cut share (10.9 % to 11.0 %), so it was not adopted; with only null move reducing the depth, ply and remaining depth almost coincide. Figures taken with the earlier Python engine have no reproducible source today and are not quoted.

## Status

Adopted; part of the reference search since the first commit (2026-10-16).
