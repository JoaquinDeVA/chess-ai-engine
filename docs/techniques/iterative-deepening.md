# Iterative deepening with a clock

> **What it is:** Searching depth 1, 2, 3... and keeping the deepest finished answer, with the rules that decide when to stop.
>
> **Read it when:** You want to know how the engine spends its time on a move.
>
> **Next:** [Transposition table](transposition-table.md)

## Problem

A game is played against a clock, not against a depth. A fixed depth wastes time in easy positions and
loses on time in sharp ones, and a search interrupted at a random moment has no answer to give.

## Idea

Search depth 1, then 2, then 3, and keep the move of the deepest iteration that *finished*. Before each
new iteration, predict how long it will take from the ones already done, and do not start it if it will
not fit. The shallow iterations are cheap and fill the [transposition table](transposition-table.md) and
the [killer moves](killer-moves.md), which speed up the deeper ones.

## How it works

Three numbers govern a move:

- **Soft budget**: 15 % of the time left, between 1.5 s and 40 s and never more than half the clock.
- **Estimate**: before every iteration after the first, the predicted duration of the next one. The
  default rule, `geometric_growth`, is the last duration times the geometric mean of the growth between
  consecutive iterations, kept between 2 and 8. If the estimate is more than 15 % over what is left of
  the budget, the next depth is not started.
- **Hard limit**: `1.5 × budget − 0.3 s`. The running iteration is interrupted there, marked truncated and
  *discarded*, unless it is the first (a move is always needed).

**Worked example.** 10 minutes on the clock: budget `min(0.15 × 600, 40) = 40 s`, hard limit
`1.5 × 40 − 0.3 = 59.7 s`.

| depth | took | elapsed | left of 40 s | estimate for the next | decision |
|---|---|---|---|---|---|
| 1 | 0.05 s | 0.05 s | 39.95 s | `0.05 × 4` (one duration: default growth) | go on |
| 2 | 0.2 s | 0.25 s | 39.75 s | `0.2 × 4.0 = 0.8 s` | go on |
| 3 | 0.9 s | 1.15 s | 38.85 s | `0.9 × √(4.0 × 4.5) = 3.8 s` | go on |
| 4 | 3.6 s | 4.75 s | 35.25 s | `3.6 × ∛(4.0 × 4.5 × 4.0) = 15.0 s` | 15.0 ≤ 1.15 × 35.25: go on |
| 5 | 15.0 s | 19.75 s | 20.25 s | `15.0 × 4.16 = 62.4 s` | 62.4 > 1.15 × 20.25 = 23.3: stop |

The engine plays the move of depth 5 after 19.75 s, half the budget. Had depth 6 been started and run
past 59.7 s, it would have been cut and thrown away, and the move of depth 5 played anyway.

> [!NOTE]
> The UCI binary ignores the increment (`winc`/`binc`): the budget comes from the time left alone.

## Implementation

`iterative_deepening` in `rust/src/search/deepening.rs` (loop, `OVERSHOOT_FACTOR = 1.15`, discard of
truncated iterations). The estimators are in `rust/src/search/iteration_time.rs`; `geometric_growth` is the
default; `fixed_factor` and `never_stop` remain for the tests. The other rules in the table below were
removed after the comparison. The budget and the hard limit are `clock_budget` in `rust/src/bin/uci.rs`;
`go movetime N` uses `N` as both. Minimum depth: `DEFAULT_MIN_DEPTH = 1`; technical cap `MAX_DEPTH = 30`
(`rust/src/search/options.rs`). The elapsed time runs from the start of the move, not of each iteration.

## Measured effect

**The defect that chose the estimator.** The original rule predicted the next iteration as `last² / previous`. With a table warmed by the previous moves the depth-1 iteration can last microseconds, the ratio between the first two iterations reached 42,166, and its square predicted hours: the search stopped at depth 2. A rule that bounds the growth of the iteration time (`geometric_growth`) replaced it. Its effect in games is not measured.

**Repeating iterations pays.** Not measured today.

## Status

Adopted; part of the reference search since the first commit (2026-10-16). `geometric_growth` default since
2026-10-21 (`d3e848c`). Minimum depth 3 from 2026-10-21 (`0cdf404`), 1 since 2026-10-31, with no default depth
cap: only the clock decides. A rule that spent the time left over the moves to go plus 75 % of the increment (2026-11-02, `542e4e0`) lost −64 ± 24 Elo in 720 games at 60 s + 1 s against the rule above (SPRT [0, 5] stopped by hand without verdict, `data/results/matches/reloj_nuevo/match.log`) and was reverted on 2026-11-03. Four clock rules (`TimeRules`, 2026-11-03) were removed on 2026-11-04: in a 300-game screening at 10 s + 0.1 s each lost to the plain clock (`half` −10.4 ± 30.6 Elo, `stab` −25.5 ± 33.6, `nodes` −80.1 ± 37.2; `data/results/matches/{half,stab,nodes}_vs_base/match.log`); `single` changed no game.
