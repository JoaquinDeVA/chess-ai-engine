# The search in Rust

> **What it is:** How the search moved from Python to Rust with identical results.
>
> **Read it when:** You want to know why the search runs in Rust.
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

The crate in `rust/` holds the rules (on top of `shakmaty`), the evaluations the search calls (`h1` and the
networks), the alpha-beta with its techniques, the opening book and the endgame tables. It is built as
two products: the Python module `chess_engine._core` (PyO3, `maturin`) for the laboratory and the tests, and the UCI
binary `rust/target/release/uci` that the window and fastchess play.

The legal moves are listed in the order shakmaty generates them (`ordered_moves` in `rust/src/board.rs`), so among moves of equal value the search plays the first of that order. Until 2026-11-09 they were sorted into the order python-chess generated them, to reproduce the Python engine's trees; that sort is gone, so equal-value ties can now resolve to a different move than in the published Python tournaments.

**Worked example.** The same position and the same depth return the same value in Python and in Rust; the chosen move can differ when two moves tie, and the time per node changes.

## Implementation

`rust/src/lib.rs` (crate), `rust/src/python.rs` (Python bindings: the evaluations), `rust/src/bin/uci.rs` (UCI binary: search, book and tables). Rules, book and
tables are tested in `rust/tests/rules.rs` (perft, end of game) and `rust/tests/knowledge.rs`; the techniques in
`rust/tests/techniques.rs`. Build: `pip install -e ".[dev]"` (module) and
`cargo build --release --manifest-path rust/Cargo.toml --bin uci` (binary).

## Measured effect

**Same results.** Replaying the 21,203 moves of the 200 recorded games of the depth-3 tournament, the core played
21,192 (99.95 %) with the same number of nodes; the 11 others differ in the last decimal of a value. Checked once,
when the core replaced the Python engine, against records that no longer exist (no command reproduces it today).

**Speed.** Not measured today: the figures taken when the core replaced the Python engine (8 to 35 times faster to depth 4, 2.0 to 2.8 plies deeper in 10 s) came from scripts that were removed in `6162db8` and from an earlier version of the core. The core has since been sped up to 928 k nodes per second with `linear_pst_terms` (`uci bench`, laptop, median of 5 runs; see the [CHANGELOG](../../CHANGELOG.md)); with `net_pst_w32`, the default since `8700e6f`, `uci bench 5` gives 1,020,805 nodes/s (laptop, 2026-11-09). On 60 other positions at depth 8 (a one-off harness, not kept) the speed-up of `67f413a` was 2.2 times (302 k to 661 k nodes/s), not the 2.58 of `uci bench`.

**Speed-up of 2026-11-10** (`252be26` to the three commits after it, same 349,598 nodes in `uci bench`; median of 5 alternating pairs of `uci bench 5`, desktop): 1,088,861 to 1,316,298 nodes/s (+21 %). Three changes: the clock is read once every 2,048 nodes instead of at every node (about +1 % alone), the moves of a node are listed and ordered in a stack array with a stable in-place sort instead of several `Vec`s (about +17 % alone, 1.19 M to 1.40 M in two runs), and the Zobrist key follows the move through shakmaty's `update_zobrist_hash` (about +4 % in 4 alternating pairs, within the noise of one machine: the runs of the same binary differ by up to 8 %). The Elo effect is not measured.

**Target CPU.** `.cargo/config.toml` builds for x86-64-v3 (AVX2, BMI2): 750 k to 788 k nodes/s (+5 %), 274,145 nodes before and after (`uci bench`, median of 5 runs, laptop, 2026-11-04). The binary does not run on CPUs older than 2013 (Haswell) or Zen 1.

**Tried without gain** (2026-11-04, `uci bench`, same session): `lto = "fat"`, `panic = "abort"` and the shared lock-free table for one thread (834 k against 921 k nodes/s) did not speed the search up; none is kept.

## Status

Adopted; since 2026-10-25. Since `6162db8` the Python search is gone and the UCI binary is the only engine.
