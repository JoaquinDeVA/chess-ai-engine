# Killer moves

> **What it is:** Remembering the moves that caused a cut-off at a depth and trying them first in sibling nodes.
>
> **Read it when:** You want to know how quiet moves are ordered before the history heuristic.
>
> **Next:** [Move ordering by MVV-LVA](mvv-lva.md)

**Problem.** Alpha-beta prunes best when the best move is tried first, and a good move in one branch
is often good in a sibling branch.

**What it does.** A move that causes a cut-off is remembered as a "killer" for its depth and tried
before the other moves in later nodes at that depth.

**How it is implemented.** `Killers::record` and the first step of `order` in
`rust/src/search/ordering.rs`, called from `Search::node`. Up to three killers are kept per depth
(`MAX_KILLERS_PER_DEPTH`), shared across iterations and moves, cleared at the start of each game. A
killer is identified by its squares and promotion, not by what it captures, as `chess.Move` does.

**Measured effect.** Depth 4, 60 positions: without killers 18,303 nodes per move against 15,988
(killers save 12.6 % of the nodes and 7.9 % of the time, 1.482 s against 1.365 s; `same as
baseline` 0.983). At depth 6 on 20 positions the baseline visited 224,896 nodes per move and the
search without killers 175,520 (baseline +28.1 % nodes; 16.562 s against 13.408 s): killers did *not*
reduce the nodes there. Same caveat as in the transposition table section: small sample, no explanation here.

**Status.** Kept; since the first commit (2026-10-16).
