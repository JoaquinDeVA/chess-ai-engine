# The search in Rust

> **What it is:** How the search moved from Python to Rust with identical results, and how much faster it is.
>
> **Read it when:** You want the speed numbers of the core.
>
> **Next:** [Several threads (Lazy SMP)](lazy-smp.md)

## Problem

In Python the search visited 3,000 to 12,000 nodes per second, so in the 3 to 10 seconds a move gets it reached
depth 4 or 5, far from the depths where a chess engine plays well. The time went into the interpreter, the
python-chess board and the evaluations, not into the algorithm.

## Idea

Keep the algorithm and change the language: the same search, node for node, in a compiled language, so that every
earlier measurement of nodes and moves still holds and only the time per node changes. Rust rather than C or C++,
which would be as fast per node but need CMake and `ctypes`: Rust builds into the Python package with PyO3 and
`maturin`, and brings `shakmaty` and `shakmaty-syzygy` for the rules and the tables.

## How it works

The crate in `rust/` holds the rules (on top of `shakmaty`), the evaluations the search calls (`h1`, the logistic
models and the network), the alpha-beta with its techniques, the opening book and the endgame tables. It is built as
two products: the Python module `chess_engine._core` (PyO3, `maturin`) for the laboratory and the tests, and the UCI
binary `rust/target/release/uci` that the window and fastchess play.

To keep the results identical, the legal moves are listed in the order python-chess generated them
(`generation_order` in `rust/src/board.rs`), so moves of equal value are chosen as before.

**Worked example.** The same position, `linear_pst_terms`, depth 4 (means of the table below): Python visits the
same nodes as Rust and returns the same move, in 3.429 s against 0.139 s. With a 10 s budget the Python engine
finished depth 4 and part of 5 (mean 4.62); the Rust core finishes 7 (mean 7.13).

## Implementation

`rust/src/lib.rs` (crate), `rust/src/python.rs` (Python bindings), `rust/src/bin/uci.rs` (UCI binary). Rules, book and
tables are tested in `rust/tests/rules.rs` (perft, end of game) and `rust/tests/knowledge.rs`; the techniques in
`rust/tests/techniques.rs`. Build: `pip install -e ".[dev]"` (module) and
`cargo build --release --manifest-path rust/Cargo.toml --bin uci` (binary).

## Measured effect

**Same results.** Replaying the 21,203 moves of the 200 recorded games of the depth-3 tournament, the core played
21,192 (99.95 %) with the same number of nodes; the 11 others differ in the last decimal of a value. Checked once,
when the core replaced the Python engine, against records that no longer exist (no command reproduces it today).

**Speed.** `scripts/bench/measure_search_speed.py --positions 200 --seconds 1,3,10 --depths 4` (removed in
`6162db8`; in the git history), 200 test positions, seed 42, a fresh table per position, one thread, books and
tables off; Python at commit `148b86d`. Seconds per move to depth 4:

| player | Python | Rust | times faster |
|---|---|---|---|
| `h1` | 7.995 | 0.226 | 35.4 |
| `linear_pst_terms` | 3.429 | 0.139 | 24.7 |
| `linear_pst` | 2.279 | 0.110 | 20.6 |
| `linear_material` | 1.335 | 0.076 | 17.6 |
| `linear_king_bucket` | 1.990 | 0.120 | 16.6 |
| `net4_pst` | 2.266 | 0.286 | 7.9 |

Mean depth completed per move in a time budget:

| player | 1 s: Python → Rust | 3 s: Python → Rust | 10 s: Python → Rust |
|---|---|---|---|
| `h1` | 2.58 → 5.29 | 3.25 → 6.04 | 4.12 → 6.93 |
| `linear_material` | 3.78 → 6.04 | 4.49 → 6.82 | 5.37 → 7.68 |
| `linear_pst` | 3.35 → 5.60 | 4.05 → 6.34 | 4.94 → 7.18 |
| `linear_pst_terms` | 3.19 → 5.43 | 3.85 → 6.26 | 4.62 → 7.13 |
| `linear_king_bucket` | 3.50 → 5.63 | 4.23 → 6.36 | 5.04 → 7.51 |
| `net4_pst` | 3.35 → 4.80 | 4.04 → 5.61 | 4.85 → 6.82 |

In 10 s the core searches 2.0 to 2.8 plies deeper. The network gained least because at the time it recomputed its
first layer at every leaf; the [incremental accumulator](incremental.md) was written afterwards for that.

## Status

Adopted; since 2026-10-25. Since `6162db8` the Python search is gone and the UCI binary is the only engine.
