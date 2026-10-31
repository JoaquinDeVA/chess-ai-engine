# History heuristic (`history`)

> **What it is:** Ordering quiet moves by how often they caused a cut-off anywhere in the search.
>
> **Read it when:** You want to know how quiet moves are ordered.
>
> **Next:** [Principal variation search](pvs.md)

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

`History` in `rust/src/search/ordering.rs` (`reward`, `score`), indexed by colour and `move_id`; `order` uses it.
The counts live for the whole game, like the table, and are cleared on `ucinewgame`. Switch: `history`.

## Measured effect

Rust core, `linear_pst_terms`, 100 positions at depth 6 (`scripts/bench/compare_search_features.py`, removed in
`6162db8`; in the git history): nodes per move 672,800 → 391,130, time 1.86 s → 1.16 s, same move as the reference
in 100 %. In games, with `tt_move`: +92 Elo ([catalogue](README.md#how-the-switches-were-judged)).

## Status

Adopted, on by default; since 2026-10-25. A malus for quiet moves that failed to cut is an idea, not measured.
