# Move ordering by MVV-LVA

> **What it is:** Trying captures first, the most valuable victim taken by the least valuable attacker first.
>
> **Read it when:** You want to know the order in which the search tries moves.
>
> **Next:** [Quiescence search](quiescence.md)

## Problem

How much alpha-beta can skip depends on the order of the moves: a refutation tried first cuts the node at
once, the same refutation tried last saves nothing. The move generator returns moves in an order that says
nothing about how good they are.

## Idea

Captures are the moves most likely to change the value, and taking a queen with a pawn is more promising than
taking a pawn with a queen. Try captures first, sorted by **M**ost **V**aluable **V**ictim, then **L**east
**V**aluable **A**ttacker.

## How it works

Each capture scores `10 × victim − attacker`, with pawn 1, knight 2, bishop 3, rook 4, queen 5, king 6; en
passant counts as a pawn taking a pawn. Quiet moves score 0 and keep their order. The full order of a node is:

1. the move stored in the transposition table ([`tt_move`](tt-move.md), if on);
2. captures by MVV-LVA score;
3. the [killer moves](killer-moves.md);
4. the other quiet moves, in generation order (until `efb595e`, by [history](history.md) score when that switch was on).

**Worked example.** A position with four captures:

| capture | victim | attacker | score | order |
|---|---|---|---|---|
| `bxc6` (pawn takes queen) | 5 | 1 | 49 | 1st |
| `Rxd5` (rook takes queen) | 5 | 4 | 46 | 2nd |
| `Nxe5` (knight takes pawn) | 1 | 2 | 8 | 3rd |
| `Qxe5` (queen takes pawn) | 1 | 5 | 5 | 4th |

If `bxc6` wins enough to reach `beta`, the other three captures and every quiet move are never searched.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../img/move-ordering-dark.svg">
  <img src="../img/move-ordering-light.svg" alt="Order in which moves are tried: table move, captures by MVV-LVA, killers, then the other quiet moves; an early cut-off skips the later moves">
</picture>

## Implementation

`mvv_lva_score` and `order` in `rust/src/search/ordering.rs`. The sort is stable, so equal captures keep the
order of generation (`ordered_moves` in `rust/src/board.rs`, the order shakmaty generates). Always on (not a switch).

## Measured effect

Not measured with the Rust engine. Figures taken with the earlier Python engine have no reproducible source today and are not quoted.

## Status

Adopted; part of the reference search since the first commit (2026-10-16).
