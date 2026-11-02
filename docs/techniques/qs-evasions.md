# Quiescence evasions (`qs_evasions`)

> **What it is:** In the quiescence search, a side in check searches every legal move instead of standing pat; implemented, off by default.
>
> **Read it when:** You wonder how the quiescence search treats checks, or why this switch is off.
>
> **Next:** [Quiescence with the table](qs-tt.md)

## Problem

The [quiescence search](quiescence.md) lets the side to move stand pat and tries only captures and promotions. A
side in check cannot stand pat: it has to answer the check, and the answer is often a quiet move. Standing pat
there scores the position as if nothing were attacked, so a fork with check at the horizon is missed.

## Idea

When the side to move is in check, do not stand pat and try every legal move (all of them are evasions). Once
the check is answered, the quiescence goes back to captures and promotions with the plies it had left.

## How it works

At a quiescence node with the side to move in check:

1. No stand-pat value: the best value starts at the worst possible for the mover, and `alpha` and `beta` are not
   narrowed by the static evaluation.
2. Every legal move is tried, with one ply less of quiescence each time.
3. Out of check, the node is a normal quiescence node.

**Worked example** (the test position). White: king g1, knight b5, pawn h2. Black: king e8, queen a8. At the
horizon White has played `Nc7+`, forking king and queen. Standing pat, the position is a knight and a pawn
against a queen: about −5 pawns for White. With evasions, Black must move the king (the queen cannot take the
knight), and White takes the queen with `Nxa8`: +4 pawns. The search now sees that `Nc7+` wins the queen.

## Implementation

`Search::quiescence` in `rust/src/search/node.rs` (the `evading` flag). Switch: `qs_evasions` (UCI option
`Techniques`, not in the default list), field `qs_evasions` of `Techniques` in `rust/src/search/options.rs`.
Test: `quiescence_evasions_see_the_fork_after_a_check` in `rust/src/search/tests.rs`. The checks given by the
side to move are not searched; only the answers to a check received.

## Measured effect

**SPRT at a fixed 20,000 nodes per move** (2026-11-02, `data/matches/qs_evasions/match.log`). The default with
`qs_evasions` (A) against the default (B), with the same search effort for both, so the clock plays no part
([How a change is measured](../MEASUREMENT.md)):

```bash
fastchess -engine cmd=rust/target/release/uci name=A option.Techniques=incremental,tt_move,history,pvs,aspiration,null_move,qs_evasions \
    -engine cmd=rust/target/release/uci name=B option.Techniques=incremental,tt_move,history,pvs,aspiration,null_move \
    -each proto=uci nodes=20000 option.Hash=16 -openings file=data/openings/openings.epd format=epd order=random \
    -sprt elo0=-5 elo1=0 alpha=0.05 beta=0.05 model=normalized -rounds 3000 -games 2 -concurrency 5 -ratinginterval 100 -srand 21
```

| test | games | score | Elo (95 %) | LLR | pentanomial | verdict |
|---|---|---|---|---|---|---|
| default + `qs_evasions` vs default, 20,000 nodes | 3,428 | 46.94 % | −21.3 ± 10.6 | −2.96 | 362, 172, 769, 136, 275 | H0: `qs_evasions` regresses |

The interval is far below 0. It costs nodes at the same budget: the test penalises a change that spends nodes.
The nodes it adds were not counted.

## Status

Implemented, off by default; rejected by the SPRT at 20,000 nodes (2026-11-02: −21.3 ± 10.6 Elo, H0).
