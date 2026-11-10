# Reverse futility pruning (`rfp`)

> **What it is:** Returning the static value of a shallow node without searching it when that value is already far beyond the bound.
>
> **Read it when:** You want to know how the search skips nodes that are clearly won or lost for the side to move.
>
> **Next:** [Search techniques](README.md)

## Problem

At a shallow node whose static value already exceeds `beta` by a wide margin, the search still tries moves
to confirm what the evaluation says. A few plies cannot change a gap that large except by a tactic.

## Idea

If the static value beats the bound by more than a few plies of tactics could change, trust it and return it.
The margin grows with the plies left: the more search remains, the more can change.

## How it works

Conditions: not at the root, not in check, at most 6 plies left, a window of no width, and both bounds outside
the range of the mates (`alpha > 0.01`, `beta < 0.99`; a static value, clipped to `0.01 .. 0.99`, cannot judge
those). The distance is measured in log-odds, `ln(p / (1 − p))`, so it means the same near 0.5 and near 0.9. The margin is
`0.46` per ply left (the 80 centipawns per ply of the plan on the usual logistic scale, `80 · ln 10 / 400`; this scale is an assumption,
not measured for these networks). The side that maximizes prunes when `static ≥ beta` and the log-odds of the static value exceed
those of `beta` by the margin; the side that minimizes, the mirror image with `alpha`.

**Worked example.** 3 plies left: margin `0.46 · 3 = 1.38`. `beta = 0.60` (log-odds 0.405), static value `0.85`
(log-odds 1.735): the distance is 1.33 < 1.38, so the node is searched. With static `0.87` (1.900) the distance is 1.49 ≥ 1.38: the node returns `0.87`
without searching a move. At 1 ply left the margin is only 0.46: `beta = 0.60` is pruned from a static value of `0.70` (0.847).

## Implementation

The branch of `Search::node` before the null move in `rust/src/search/node.rs` (`RFP_UP_TO_DEPTH`, `RFP_MARGIN_PER_PLY`,
`log_odds`). Switch `rfp`. Tests in `rust/src/search/tests.rs`: the forced mate and the hanging queen are unchanged,
and the margin means the same at 0.5 and at 0.9. A first version without the mate-range guard found a slower mate than the
shortest one (0.9995 instead of 0.9997); the guard fixes it.

## Measured effect

`TTD_TECHNIQUES=<set> ./target/release/uci ttd 11 bench/middlegame.epd 60` from `rust/`, 1 thread, `net_pst_w32`, 20 positions,
desktop, 2026-11-10, the default set with `lmr` with and without `rfp`: 324,595,891 to 215,120,644 nodes (−33.7 %), median
11.13 s to 8.58 s, branching factor 2.75 to 2.66; nodes per second fall from 1.20 M to 1.03 M (more static evaluations). Elo effect: not measured, SPRT queued for the node.

## Status

On by default since 2026-11-10, pending its SPRT [−5, 0] on the node.
