# Futility pruning (`futility`)

> **What it is:** At a shallow depth, skipping the quiet moves of a node whose static value is so far below the best value so far that a quiet move cannot close the gap.
>
> **Read it when:** You want to know how the search drops hopeless quiet moves near the horizon.
>
> **Next:** [Late move pruning](lmp.md)

## Problem

Near the horizon a quiet move changes the value by little. If the position is already much worse than what
the side to move can get elsewhere, searching its quiet moves only confirms that.

## Idea

Compare the static value, plus the most a quiet move could gain in the plies left, with the bound the
side already has. If it falls short, skip the quiet moves. It is the mirror of [reverse futility pruning](rfp.md):
that one cuts nodes that are far above the bound, this one moves that are far below it.

## How it works

Conditions: not in check, at most 3 plies left, `alpha > 0.01` and `beta < 0.99`, and the move is not the first one,
a capture, a promotion, a killer or a move that gives check. The margin is `0.69` log-odds per ply left
(120 centipawns per ply on the usual logistic scale, `120 · ln 10 / 400`; an assumption, not measured for these networks).
The side that maximizes skips a move when `log_odds(alpha) − log_odds(static) ≥ margin`; the side that minimizes, when
`log_odds(static) − log_odds(beta) ≥ margin`.

**Worked example.** 2 plies left, margin `0.69 · 2 = 1.38`. The maximizing side has `alpha = 0.60` (log-odds 0.405) and the
static value is `0.20` (−1.386): the distance is 1.79 ≥ 1.38, so its quiet moves (except the first, the killers and the checks) are skipped.
With a static value of `0.35` (−0.619) the distance is 1.02 < 1.38: they are searched.

## Implementation

The `futility` check in the move loop of `Search::node`, `rust/src/search/node.rs` (`SHALLOW_UP_TO_DEPTH`,
`FUTILITY_MARGIN_PER_PLY`). Switch `futility`. Test: `shallow_pruning_keeps_the_forced_mate_and_the_hanging_queen`
in `rust/src/search/tests.rs`.

## Measured effect

`TTD_TECHNIQUES=<set> ./target/release/uci ttd 11 bench/middlegame.epd 60` from `rust/`, 1 thread, `net_pst_w32`, 2026-11-10, the default set
(`incremental,tt_move,null_move,see,lmr,rfp`) with and without `futility`: 215,120,644 to 144,499,072 nodes (−32.8 %), median 8.65 s to 6.96 s. Elo
effect: not measured, SPRT queued for the node.

## Status

Implemented, **off by default** pending its SPRT [−5, 0] on the node (2026-11-10).
