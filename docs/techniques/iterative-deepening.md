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

### Rules on top of the budget (`TimeRules`)

The UCI option `TimeRules` (a list of names, empty by default) adds four rules to the loop above:

- `single`: a position with one legal move is played after the first iteration.
- `half`: no iteration starts once half of the soft budget is spent.
- `stab`: the budget is multiplied by `1.20 − 0.04 n` after `n` consecutive iterations with the same best move (`n` up to 10).
- `nodes`: the budget is multiplied by `max(0.5, 2 (1 − f) + 0.4)`, where `f` is the share of the nodes of the last iteration that the best root move took.

The constants are those of Ethereal's time management, taken as a starting point and not verified against its source.

## Implementation

`iterative_deepening` in `rust/src/search/deepening.rs` (loop, `OVERSHOOT_FACTOR = 1.15`, discard of
truncated iterations). The estimators are in `rust/src/search/estimators.rs`; `geometric_growth` is the
default; `fixed_factor` and `never_stop` remain for the tests. The other rules in the table below were
removed after the comparison. The budget and the hard limit are `clock_budget` in `rust/src/bin/uci.rs`;
`go movetime N` uses `N` as both. `TimeRules` is `time_rules` in `Options` (`TimeRules::scale` and the checks in `deepen`, `rust/src/search/deepening.rs`; the share `f` is `SearchResult::best_fraction`). Minimum depth: `DEFAULT_MIN_DEPTH = 1`; technical cap `MAX_DEPTH = 30`
(`rust/src/search/options.rs`). The elapsed time runs from the start of the move, not of each iteration.

## Measured effect

All figures below were measured with the earlier Python engine.

**The defect that chose the estimator.** In the per-move logs of a 10 min + 5 s tournament (1,051 moves
of three models), 126 moves (12.0 %) ended below depth 3, all in under 2 s with a budget of 35-40 s. The
original rule predicted `last² / previous`. With a table warmed by the previous moves the depth-1 iteration
can last microseconds, the ratio between the first two iterations reached 42,166, and its square predicted
hours: the search stopped at depth 2.

**The estimators compared.** Iteration times of 85 moves from three games were recorded with the stopping
rules off (`scripts/bench/record_iteration_times.py`, removed in `6162db8`; in the git history), and the loop
was replayed on them with each rule (`scripts/bench/compare_time_estimators.py`, same). With a 40 s budget:

| rule | mean depth | budget used | stopped though the next iteration fitted | iteration cut by the hard limit |
|---|---|---|---|---|
| `ratio_squared` (original) | 4.55 | 61 % | 21 % | 9 % |
| `geometric_growth` (default) | 4.72 | 68 % | 8 % | 8 % |
| `clamped_growth` | 4.72 | 79 % | 9 % | 16 % |
| `node_growth` | 4.73 | 76 % | 7 % | 14 % |
| `fixed_factor` (×4) | 4.80 | 85 % | 1 % | 16 % |

All bounded rules are about 0.2 plies deeper than the original; the differences among them are within the
noise of 85 moves. `geometric_growth` was chosen for the fewest wasted iterations at 40 s and the lowest
overrun at 10 s (2 %). The replay was not confirmed in games.

**Repeating iterations pays.** On a middlegame position the iterations up to depth 7 took 730,876 nodes in
total against 1,083,355 for searching depth 7 directly (ad-hoc measurement, not versioned), because each
iteration seeds the move ordering of the next.

## Status

Adopted; part of the reference search since the first commit (2026-10-16). `geometric_growth` default since
2026-10-21 (`d3e848c`). Minimum depth 3 from 2026-10-21 (`0cdf404`), 1 since 2026-10-31, with no default depth
cap: only the clock decides. A rule that spent the time left over the moves to go plus 75 % of the increment (2026-11-02, `542e4e0`) lost −64 ± 24 Elo in 720 games at 60 s + 1 s against the rule above (SPRT [0, 5] stopped by hand without verdict, `data/matches/reloj_nuevo/match.log`) and was reverted on 2026-11-03. `TimeRules` (2026-11-03): implemented, off. Screening of each rule alone against the same binary, 300 games at 10 s + 0.1 s (`data/matches/{half,stab,nodes}_vs_base/match.log`, `~/worker/cribar.sh`): `half` −10.4 ± 30.6 Elo (48.50 %), `stab` −25.5 ± 33.6 (46.33 %), `nodes` −80.1 ± 37.2 (38.67 %); none passed, so none went to an SPRT. `single` changes no game and has only a unit test (`a_single_legal_move_is_played_after_the_first_iteration`); the other tests are `half_the_budget_stops_the_next_iteration` and `the_stability_and_node_factors_scale_the_budget`.
