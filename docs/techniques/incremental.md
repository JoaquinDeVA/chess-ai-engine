# Incremental network accumulator (`incremental`)

> **What it is:** Updating the first layer of the network move by move instead of recomputing it at every leaf.
>
> **Read it when:** You want to know why `net4_pst` evaluates faster in the search.
>
> **Next:** [Table move first](tt-move.md)

**Problem.** `net4_pst` (768 -> 256 -> 32 -> 32 -> 1) spent most of its time in the first layer, summing the
rows of every piece again at each leaf.

**What it does.** Keeps the first-layer sum updated move by move: a move adds and removes only the rows of the
pieces that change. The features are relative to the side to move, so two accumulators are kept (White and
Black as the reference) and the one of the player to move is read.

**How it is implemented.** `NetState` in `rust/src/eval/net.rs` behind the `EvalState` trait (`apply`,
`apply_null`, `undo`, `evaluate`), with a stack of snapshots so `undo` is a pop. The search holds the state
beside `GameState`. Switch: `incremental`. Test: `incremental_equals_full` (same values as the full evaluation up
to rounding).

**Measured effect.** About 15 % more nodes per second for `net4_pst`
(`python (removed) scripts/bench/measure_search_speed.py --techniques incremental`). Rounding can break a tie, so node counts
agree within 1 %, not exactly.

**Status.** On by default; since 2026-10-25.
