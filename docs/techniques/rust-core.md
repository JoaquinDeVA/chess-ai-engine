# The search in Rust

> **What it is:** How the search moved from Python to Rust with identical results, and how much faster it is.
>
> **Read it when:** You want the speed numbers of the core.
>
> **Next:** [Several threads (Lazy SMP)](lazy-smp.md)

**Problem.** In Python the search visited 3,000 to 12,000 nodes per second, so in the 3 to 10
seconds a move gets it reached depth 4 or 5, far from the depths where a chess engine plays well.
The time went into the interpreter, the board of python-chess and the evaluations, not into the
algorithm.

**What it does.** The same search, in Rust: the rules of the game, the evaluations the search calls
(`h1`, the four logistic models and the sparse network), the alpha-beta with all the techniques
above, the opening book and the endgame tables. Python keeps the game loop, the bots' interface, the
training and the measurement tools, and calls the core once per move with the position, the
evaluation and the time limits.

**How it is implemented.** The crate in `rust/` is built as the Python module `chess_engine._core`
(PyO3 and `maturin`) on top of `shakmaty` for the rules. Nothing changed in the algorithm: the Rust
core reproduced the Python engine node for node when it replaced it (checked then against files the Python engine
produced; they were removed afterwards and live in git history). Today `rust/tests/rules.rs` (perft, end of game) and
`rust/tests/knowledge.rs` check the rules, the book and the tables. The legal moves are listed in the order python-chess generates them
(`generation_order`), so that moves of equal value are chosen as before. Replaying the 21,203 moves
of the 200 recorded games of the depth-3 tournament, the core plays 21,192 of them (99.95 %) with
the same number of nodes; the 11 others differ in the last decimal of a value. This was checked once, when the core replaced
the Python engine, against game records that no longer exist (no command reproduces the figure today).

**Measured effect.** Measured with `python (removed) scripts/bench/measure_search_speed.py --positions 200 --seconds 1,3,10 --depths 4`
(200 test positions, seed 42, a fresh table per position, five processes of one thread each, books
and tables off), once with the Python engine (commit `148b86d`) and once with the core. Time to search
every position to depth 4, in seconds per move:

| player | Python | Rust | times faster |
|---|---|---|---|
| Heuristic (`h1`) | 7.995 | 0.226 | 35.4 |
| Position-terms logistic model (`linear_pst_terms`) | 3.429 | 0.139 | 24.7 |
| Piece-square logistic model (`linear_pst`) | 2.279 | 0.110 | 20.6 |
| Material logistic model (`linear_material`) | 1.335 | 0.076 | 17.6 |
| King-bucket logistic model (`linear_king_bucket`) | 1.990 | 0.120 | 16.6 |
| Four-layer piece-square network (`net4_pst`) | 2.266 | 0.286 | 7.9 |

Mean depth completed in a time budget per move (the clock decides; the estimator stops an iteration
that will not fit, so a move uses 50 to 80 % of the budget):

| player | 1 s: Python → Rust | 3 s: Python → Rust | 10 s: Python → Rust |
|---|---|---|---|
| `h1` | 2.58 → 5.29 | 3.25 → 6.04 | 4.12 → 6.93 |
| `linear_material` | 3.78 → 6.04 | 4.49 → 6.82 | 5.37 → 7.68 |
| `linear_pst` | 3.35 → 5.60 | 4.05 → 6.34 | 4.94 → 7.18 |
| `linear_pst_terms` | 3.19 → 5.43 | 3.85 → 6.26 | 4.62 → 7.13 |
| `linear_king_bucket` | 3.50 → 5.63 | 4.23 → 6.36 | 5.04 → 7.51 |
| `net4_pst` | 3.35 → 4.80 | 4.04 → 5.61 | 4.85 → 6.82 |

In 10 seconds the core searches 2.0 to 2.8 plies deeper. A 200-game round robin at a fixed depth of 3
took 56 s (it took hours in Python). The network gains least because it still adds the 768 inputs
to its first layer at every leaf; an accumulator updated move by move, as NNUE does, would remove
that and has not been written.

**Status.** Kept; since 2026-10-25.
