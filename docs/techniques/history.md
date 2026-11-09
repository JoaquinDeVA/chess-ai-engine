# History heuristic (`history`)

> **What it is:** Ordering quiet moves by how often they caused a cut-off anywhere in the search.
>
> **Read it when:** You want to know how quiet moves are ordered.
>
> **Next:** [History malus (retired)](history-malus.md)

## Problem

Captures are ordered by [MVV-LVA](mvv-lva.md), but among quiet moves nothing says which to try first;
[killers](killer-moves.md) remember only three per depth.

## Idea

A quiet move that refuted one position often refutes others: the same rook to the open file, the same knight jump.
Count, for every move, how often it caused a cut-off, weighting deep cut-offs more because they saved more work,
and try quiet moves with high counts first.

## How it works

When a quiet move causes a cut-off with `d` plies left, its count for the side that played it grows by `d²`.
Quiet moves that are not killers are ordered by count, highest first. When a count passes 2²⁴, all counts are
halved, so old successes fade.

**Worked example.** `Rd1` for White cut at depth 4 (+16) and at depth 2 (+4): count 20. `h3` never cut: 0.
In a later node the quiet moves `h3`, `a3`, `Rd1` (generation order) are tried as `Rd1`, then `h3`, `a3`.

## Implementation

Removed in `efb595e` (2026-11-09) with its switch, its tests and its UCI name; the last version is in `a1bdf60`. It lived in `History` in `rust/src/search/ordering.rs` (`reward`, `score`), indexed by colour and `move_id`, and `order`
used it. The counts lived for the whole game, like the table, and were cleared on `ucinewgame`. Switch: `history`.
Killer moves never depended on it.

## Measured effect

Rust core, `linear_pst_terms`, 100 positions at depth 6 (`scripts/bench/compare_search_features.py`, removed in
`6162db8`; in the git history): nodes per move 672,800 → 391,130, time 1.86 s → 1.16 s, same move as the reference
in 100 %. In games, with `tt_move`: +92 Elo ([catalogue](README.md#how-the-switches-were-judged)).

**SPRT [−5, 0]**, the default set against the same set without `history`. Round robin of 2026-11-06 to 2026-11-09 (fastchess, 10 s + 0.1 s, `Hash` 16, the 10,000 openings of `data/openings/openings.epd`, the model then embedded, `linear_pst_terms`; queue `src/chess_lab/scripts/worker/queue.json` at `d2d03ba`): +17.5 ± 12.4 Elo at 2,200 games,
LLR +2.09 of ±2.94, no verdict when the round robin was stopped (`data/results/roundrobin/abl_history/llr.csv`).

## Status

Retired, to be taken up again (`efb595e`, 2026-11-09), together with its [malus](history-malus.md). Adopted on
2026-10-25 before the SPRT existed; its SPRT gave no verdict, and the core keeps only switches accepted by one.
To take it up again: restore it from `a1bdf60` and run its SPRT to a verdict against the current default ([How a change is measured](../MEASUREMENT.md)).
