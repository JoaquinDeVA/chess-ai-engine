# Aspiration windows (`aspiration`)

> **What it is:** Starting each iteration with a narrow window around the previous value.
>
> **Read it when:** You want to know how an iteration reuses the value of the previous one.
>
> **Next:** [Null move](null-move.md)

**Problem.** Each iteration starts from the whole `[0.01, 0.99]` window although the previous iteration's value
is almost always close.

**What it does.** From depth 4, searches with a window of +-0.02 around the last value; if the result falls
outside, widens it by 4 times and searches again.

**How it is implemented.** `iteration` in `rust/src/search/deepening.rs`. A repeat cut short by the clock keeps the
last complete result, and nothing reduced is stored as complete in the table.

**Measured effect.** Nodes 672,800 -> 440,213; same move 100 %. With `pvs`: +104 Elo (with `tt_move`, `history`).

**Status.** On by default.
