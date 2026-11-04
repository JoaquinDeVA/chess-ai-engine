# History malus (`history_malus`)

> **What it is:** Lowering the history score of the quiet moves that were tried before the one that caused a cut-off.
>
> **Read it when:** You wonder why quiet moves that fail are not demoted in the move order, or why this switch is off.
>
> **Next:** [Search techniques](README.md)

## Problem

The [history heuristic](history.md) only rewards: a quiet move that cuts gains score. A quiet move that is tried first, fails, and is followed by a cut from another move keeps its old score, so it is tried early again in the sibling nodes.

## Idea

When a quiet move causes a cut-off, the quiet moves tried before it were wrong guesses. Give them the opposite of the reward, so the order learns from failures as well as from successes.

## How it works

At a cut-off with `d` plies left, the cutting quiet move gains `d²` (as in `history`) and every quiet move tried before it in that node loses `d²`. Captures and promotions are neither rewarded nor penalised. Scores can go negative; when one passes 2²⁴ in absolute value, all are halved.

**Worked example.** At depth 3 the quiet moves `h3`, `a3`, `Rd1` are tried in this order and `Rd1` cuts. `Rd1` gains 9; `h3` and `a3` lose 9 each. In the next node `Rd1` comes first and `h3`, `a3` last.

## Implementation

`History::penalise` in `rust/src/search/ordering.rs`; the list of failed quiet moves in `Search::node` (`rust/src/search/node.rs`). Switch: `history_malus` (UCI option `Techniques`, field `history_malus` of `Techniques`); it has effect only together with `history`. Test: `the_history_malus_does_not_change_the_value` in `rust/tests/techniques.rs`.

## Measured effect

SPRT [0, 5] at a fixed 20,000 nodes per move, `incremental,tt_move,history,pvs,aspiration,null_move,see,history_malus` against the same list without it (`data/matches/history_malus_nodes/match.log`, 4 games at a time): stopped by hand at 2,404 reported games (of 2,556 played), +3.9 ± 9.3 Elo, 50.56 % of the points, LLR +0.33 of ±2.94. No verdict; the estimate fell from +31 ± 33 at 202 games to nearly zero, so any effect is small. Not measured with the clock.

## Status

Implemented, off by default; no measured gain (2026-11-04).
