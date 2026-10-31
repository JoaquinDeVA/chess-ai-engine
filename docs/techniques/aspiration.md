# Aspiration windows (`aspiration`)

> **What it is:** Starting each iteration with a narrow window around the previous iteration's value.
>
> **Read it when:** You want to know why an iteration is sometimes searched twice.
>
> **Next:** [Null move](null-move.md)

## Problem

Each iteration of [iterative deepening](iterative-deepening.md) starts with the whole window, although the value
of the previous iteration is almost always close to the new one. A wide window cuts little.

## Idea

Guess that the value will stay near the last one and search with a narrow window around it. If the guess is right
the search is cheaper; if the value falls outside, widen the window and search again.

## How it works

From depth 4, iteration `n` starts with `(v − 0.02, v + 0.02)`, where `v` is the value of iteration `n − 1`. If the
result falls at or below the lower edge, that edge moves down; at or above the upper edge, that edge moves up. Each
repeat multiplies the half-width by 4 (0.02, 0.08, 0.32); past 0.4 the window becomes the whole interval.

**Worked example** (illustrative values). Depth 5 gave `0.55`. Depth 6 searches `(0.53, 0.57)` and returns `0.50` (fails low). The
lower edge moves to `0.55 − 0.08 = 0.47`: `(0.47, 0.57)`, which returns `0.50`, inside: done, two searches. A
result that kept falling outside would be searched at most four times, the last with the whole window.

## Implementation

`iteration` in `rust/src/search/deepening.rs`, with `ASPIRATION_FROM_DEPTH = 4`, `ASPIRATION_DELTA = 0.02` and
`ASPIRATION_LIMIT = 0.4`. A repeat cut short by the clock ends the iteration and keeps the last complete result; a
search outside its window is never stored as exact in the table. Switch: `aspiration`.

## Measured effect

Rust core, `linear_pst_terms`, 100 positions at depth 6 (`scripts/bench/compare_search_features.py`, removed in
`6162db8`; in the git history): nodes per move 672,800 → 440,213, time 1.86 s → 1.20 s, same move in 100 %. In
games, with `pvs` (and `tt_move`, `history`): +104 Elo ([catalogue](README.md#how-the-switches-were-judged)).

## Status

Adopted, on by default; since 2026-10-25.
