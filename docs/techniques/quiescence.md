# Quiescence search

> **What it is:** Continuing captures and promotions past the depth limit so a position is not scored in the middle of an exchange.
>
> **Read it when:** You want to know how the horizon effect is handled.
>
> **Next:** [Make/unmake and fast terminal detection](board-state.md)

**Problem.** A fixed-depth search ends in the middle of an exchange: it sees "queen takes pawn", misses
the recapture and scores a free pawn (the horizon effect).

**What it does.** At depth 0 the position is not scored directly. The side to move may keep the static
evaluation ("stand-pat": nobody is forced to capture) or follow a *noisy* move (capture or promotion)
for a few more plies, with alpha-beta pruning, and the best option is returned.

**How it is implemented.** `Search::quiescence` in `rust/src/search/node.rs`, called from
`Search::node` at `depth_remaining <= 0`, and `is_noisy` in `rust/src/board.rs` (legal captures and
promotions). It uses neither the table nor killers; its nodes count in `nodes_visited`. The bots
(`SearchBot`) use `DEFAULT_QUIESCENCE_PLIES = 4`; `quiescence_plies=0` switches it off, which
is also the default of the search function itself.

**Measured effect.**

Measured with `python (removed) scripts/bench/compare_quiescence.py --players h1,linear_pst_terms` on 1000
test positions (seed 42); agreement is the share of positions where the chosen move is Stockfish's
move (report: `data/reports/QUIESCENCE.md`).

| player | depth | quiescence | agreement | nodes/move | s/move |
|---|---|---|---|---|---|
| Heuristic (`h1`) | 3 | off | 0.308 | 1,648 | 0.392 |
| Heuristic (`h1`) | 3 | 4 plies | 0.355 | 5,980 | 1.454 |
| Position-terms logistic model (`linear_pst_terms`) | 3 | off | 0.291 | 1,601 | 0.209 |
| Position-terms logistic model (`linear_pst_terms`) | 3 | 4 plies | 0.387 | 3,974 | 0.571 |

Paired difference at depth 3 (quiescence minus none): +0.047 for `h1` and +0.096 for
`linear_pst_terms`, both with a standard error of 0.015 (three standard errors or more). The price is
2.5 to 3.6 times the nodes and about 2.7 to 3.7 times the time per move at the same depth.

**Status.** Kept and on by default in the search bots; added on 2026-10-17 (commit `40571a4`).
