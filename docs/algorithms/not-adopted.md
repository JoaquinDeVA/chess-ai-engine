# Techniques not adopted

> **What it is:** Two ideas that were tried and are not in use: late move reductions (implemented, off) and tactical move ordering with Prolog (removed).
>
> **Read it when:** You wonder why an obvious technique is missing.
>
> **Next:** [Data and training](../DATA_AND_TRAINING.md)

## Late move reductions (`lmr`) - tried, not adopted

**Problem.** The last quiet moves of an ordered list are rarely best but cost as much as the first.

**What it does.** From the 5th move (15th for a deeper cut) at depth 4 or more, quiet moves are searched one ply
less, and again in full if they beat alpha.

**How it is implemented.** `child_value` with `LMR_FROM_MOVE`, `LMR_FROM_DEPTH`, `LMR_DEEPER_FROM_MOVE`.

**Measured effect.** Aggressive: 86,830 nodes but the same move as the reference in only 70 %, and 63.5 % in games
with all techniques (below the 76.5 % without it). Mild: 49.0 % against the set without it (-7 Elo) although it
searched half a ply deeper.

**Status.** Implemented, off by default; rejected on the evidence of games. Re-measure it with a stronger
evaluation, which orders moves better and would make reductions safer.

## Tried and removed: tactical move ordering with Prolog

The repository once had a `TacticalAlphaBetaBot` that asked a Prolog knowledge base which legal moves
create a tactical pattern (forks, pins, skewers) and searched those first, hoping for earlier
cut-offs. Commit `62c2904` measured about 2 ms per node (every node loaded its position into Prolog),
several seconds per move at depth 3 against about one second without it, and sometimes more nodes.
Commit `ca35218` (2026-10-18) removed it because no bot or script used it. In one sentence: the idea
was never shown to pay for its cost, and it was dropped rather than disproved.
