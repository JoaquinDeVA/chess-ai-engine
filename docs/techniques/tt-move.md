# Table move first (`tt_move`)

> **What it is:** Trying the best move stored in the transposition table before any other.
>
> **Read it when:** You want to know the biggest saving in nodes among the switches.
>
> **Next:** [History heuristic](history.md)

**Problem.** A node whose table entry is too shallow to cut still knows the best move found there; the reference
ignores it and orders by MVV-LVA and killers.

**What it does.** Tries the stored best move before every other move.

**How it is implemented.** `move_to_front` in `rust/src/search/ordering.rs`, called in `Search::node`. The value does
not change (same tree, other order), only how soon a cut-off comes.

**Measured effect.** Nodes at depth 6 fall from 672,800 to 256,693 (2.6 times); same move as the reference in 98 %
(ties resolve differently). In games, with `history`: +92 Elo.

**Status.** On by default.
