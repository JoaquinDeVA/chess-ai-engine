# History heuristic (`history`)

> **What it is:** Ordering quiet moves by how often they caused a cut-off before.
>
> **Read it when:** You want to know how quiet moves are ordered.
>
> **Next:** [Principal variation search](pvs.md)

**Problem.** Among quiet moves nothing says which to try first; killers only remember two or three per depth.

**What it does.** Counts, for each (colour, move), how often it produced a cut-off, weighted by depth, and orders
quiet moves by that count.

**How it is implemented.** `History` in `rust/src/search/ordering.rs`, indexed by `move_id`; `reward` is called at a
cut-off by a quiet move. Part of `order`.

**Measured effect.** Nodes 672,800 -> 391,130; same move as the reference 100 %.

**Status.** On by default.
