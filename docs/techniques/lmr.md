# Late move reductions (`lmr`)

> **What it is:** Searching the late quiet moves of an ordered list one ply less; implemented, off.
>
> **Read it when:** You wonder why an obvious technique is off.
>
> **Next:** [Tactical move ordering with Prolog](prolog-ordering.md)

**Problem.** The last quiet moves of an ordered list are rarely best but cost as much as the first.

**What it does.** From the 5th move (15th for a deeper cut) at depth 4 or more, quiet moves are searched one ply
less, and again in full if they beat alpha.

**How it is implemented.** `child_value` with `LMR_FROM_MOVE`, `LMR_FROM_DEPTH`, `LMR_DEEPER_FROM_MOVE`.

**Measured effect.** Aggressive: 86,830 nodes but the same move as the reference in only 70 %, and 63.5 % in games
with all techniques (below the 76.5 % without it). Mild: 49.0 % against the set without it (-7 Elo) although it
searched half a ply deeper.

**Status.** Implemented, off by default; rejected on the evidence of games. Re-measure it with a stronger
evaluation, which orders moves better and would make reductions safer.
