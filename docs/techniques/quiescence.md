# Quiescence search

> **What it is:** Continuing captures and promotions past the depth limit so a position is not scored in the middle of an exchange.
>
> **Read it when:** You want to know how the horizon effect is handled.
>
> **Next:** [Make/unmake and fast terminal detection](board-state.md)

## Problem

A fixed-depth search ends in the middle of an exchange: it sees "queen takes pawn" as its last move, misses
the recapture one ply later and scores a free pawn. This is the horizon effect.

## Idea

At the depth limit, do not score the position at once. Let the side to move either accept the static
evaluation ("stand pat": nobody is forced to capture) or play a capture or promotion, and continue that way
for a few more plies until the position is quiet.

## How it works

At `depth_remaining = 0` the node becomes a quiescence node:

1. The static evaluation is the *stand-pat* value. If it already reaches `beta`, return it (cut).
2. Otherwise raise `alpha` to it, and try each capture and promotion, with alpha-beta, one ply less of
   quiescence each time.
3. After 4 plies of quiescence the position is scored statically whatever is left.

**Worked example.** Values in pawns for White, for readability. White's last move at depth 0 is `Qxd5`,
taking a pawn defended by the knight on c7. Without quiescence the position scores `+1`. With it, Black to move:

| Black's option | value for White |
|---|---|
| stand pat (do nothing) | `+1` |
| `Nxd5` (knight takes queen) | `+1 − 9 = −8`; White then has no capture: stand pat |

Black takes the minimum, `−8`: the search now sees that `Qxd5` loses the queen and picks another move.

## Implementation

`Search::quiescence` in `rust/src/search/node.rs`, called from `Search::node` at `depth_remaining ≤ 0`, and
`is_noisy` in `rust/src/board.rs` (legal captures and promotions). It uses neither the transposition table nor
the killers; its nodes count in `nodes_visited`. `DEFAULT_QUIESCENCE_PLIES = 4` (`rust/src/search/options.rs`),
used by the UCI binary; the Python bindings take `quiescence_plies`, where 0 switches it off.

## Measured effect

Measured with the earlier Python engine on 1,000 test positions (seed 42), depth 3
(`scripts/bench/compare_quiescence.py --players h1,linear_pst_terms`, removed in `6162db8`; in the git history;
report `data/reports/QUIESCENCE.md`). Agreement is the share of positions where the move is Stockfish's:

| player | quiescence | agreement | nodes per move | s per move |
|---|---|---|---|---|
| `h1` | off | 0.308 | 1,648 | 0.392 |
| `h1` | 4 plies | 0.355 | 5,980 | 1.454 |
| `linear_pst_terms` | off | 0.291 | 1,601 | 0.209 |
| `linear_pst_terms` | 4 plies | 0.387 | 3,974 | 0.571 |

Paired differences: +0.047 for `h1` and +0.096 for `linear_pst_terms`, standard error 0.015 each. The price
is 2.5 to 3.6 times the nodes at the same depth.

## Status

Adopted; on since 2026-10-17 (`40571a4`).
