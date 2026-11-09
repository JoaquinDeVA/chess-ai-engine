# Table move first (`tt_move`)

> **What it is:** Trying first the best move the transposition table remembers for the position.
>
> **Read it when:** You want to know which technique saves the most nodes.
>
> **Next:** [Null move](null-move.md)

## Problem

An entry of the [transposition table](transposition-table.md) that is too shallow to give a value still knows
which move was best when the position was searched before. The reference search ignored it and ordered by
[MVV-LVA](mvv-lva.md) and [killers](killer-moves.md) only, so a cut-off that one move would give came late.

## Idea

The move that was best one iteration ago is usually still best. Search it first: alpha-beta cuts soonest when the
best move comes first.

## How it works

Before the moves of a node are searched, the stored best move (if any) is moved to the front; the rest keep their
order. The value of the node does not change, only how soon a cut-off comes.

**Worked example** (illustrative). At depth 6, a node was searched at depth 5 in the previous iteration and its entry says
`Bb5`. The order by MVV-LVA and killers would try `Bb5` 12th of 30 moves. With `tt_move` it is tried first; it
reaches beta again, and the 29 other moves are not searched. Without it, eleven moves are searched to depth 5
before the cut.

## Implementation

`move_to_front` in `rust/src/search/ordering.rs`, called in `Search::node` (`rust/src/search/node.rs`). Helper
threads of [Lazy SMP](lazy-smp.md) skip it at the root, where they rotate the moves on purpose. Switch: `tt_move`.

## Measured effect

Rust core, `linear_pst_terms`, 100 positions at depth 6
(`scripts/bench/compare_search_features.py`, removed in `6162db8`; in the git history): nodes per move fall from
672,800 to 256,693 (2.6 times) and time from 1.86 s to 0.71 s; the move equals the reference's in 98 % (ties are
resolved in another order). In games, together with `history`: 63.0 %, +92 Elo against the reference
(table in the [catalogue](README.md#how-the-switches-were-judged)).

**SPRT [−5, 0]**, the default set of then (`incremental,tt_move,history,pvs,aspiration,null_move,see`) against the same set without `tt_move`. Round robin of 2026-11-06 to 2026-11-09 (fastchess, 10 s + 0.1 s, `Hash` 16, the 10,000 openings of `data/openings/openings.epd`, the model then embedded, `linear_pst_terms`; queue `src/chess_lab/scripts/worker/queue.json` at `d2d03ba`): +56.5 ± 18.3 Elo at 1,098 games, LLR +2.96 of ±2.94, **H1 accepted**: removing it loses (`data/results/roundrobin/abl_tt_move/llr.csv`). Measured before `pvs`, `aspiration` and `history` were retired; the set without them has no SPRT of its own yet.

## Status

Adopted, on by default; since 2026-10-25. Kept in the core on 2026-11-09 (`efb595e`): its removal SPRT accepted H1.
