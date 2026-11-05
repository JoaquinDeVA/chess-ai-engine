# Several threads (Lazy SMP)

> **What it is:** Running the search on several threads that share one lock-free transposition table.
>
> **Read it when:** You want to use several threads or know what they gain.
>
> **Next:** [Incremental network accumulator](incremental.md)

## Problem

A computer has several cores and the search used one.

## Idea

Let every thread search the same root and share one transposition table. The helpers do not need to coordinate:
what one finds is in the table when another reaches the same position. Dividing the root moves among the threads
was not chosen: the first move takes most of the work (the others are cut off by its value), so the threads would
wait for each other and, blind to each other's results, prune less.

## How it works

- The main thread runs [iterative deepening](iterative-deepening.md) as usual, with the clock.
- Each helper runs its own iterative deepening, with aspiration windows, on its own copy of the position. It never
  searches shallower than the main thread is, and it skips depths by a pattern that depends on its index (skip
  size 1, 1, 2, 2, 2, 2 and phase 0, 1, 0, 1, 2, 3 for helpers 1 to 6), so the helpers do not all search the same depth.
  It also rotates the moves of the root by its index.
- The table and the history scores are shared without locks; a table slot written by two threads at once is detected
  and ignored. Each thread keeps its own killer moves.
- When the main thread finishes it raises a stop flag and the helpers abandon their search.
- The move is chosen by a vote: every thread whose last finished iteration is at least as deep as the main thread's
  gives its move a weight of its depth times (its value minus the lowest value among the voters plus 0.01). The move
  with most weight is played and the main thread's move wins a tie. With one thread there is no vote.

**Worked example.** Four threads; the main thread is on depth 6. Helper 1 skips odd depths, so it searches 6 and then 8; helper 3 skips depths 6 and 7 and searches 8. When helper 2 stores an exact value for a position at depth 5 from the root, the main thread, reaching that position with one ply left, finds an entry deep enough and returns without searching it.

**Vote example.** The figure shows four threads that have finished depth 8 or more (the lowest value, 0.50, is helper 3's, so the margin is 0.50 − 0.01):

| thread | move | depth | value | weight = depth × (value − 0.50 + 0.01) |
|---|---|---|---|---|
| main | e4 | 8 | 0.52 | 0.24 |
| helper 1 | d4 | 8 | 0.55 | 0.48 |
| helper 2 | d4 | 9 | 0.58 | 0.81 |
| helper 3 | e4 | 8 | 0.50 | 0.08 |

e4 gets 0.32 and d4 gets 1.29, so d4 is played although the main thread chose e4. A helper whose last finished iteration is shallower than the main thread's does not vote.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../img/threads-shared-table-dark.svg">
  <img src="../img/threads-shared-table-light.svg" alt="Lazy SMP: the depths each thread searches, what the threads share and a vote among four threads">
</picture>

## Implementation

`iterative_deepening`, `helper` and `vote` in `rust/src/search/deepening.rs`; `SharedTable` in
`rust/src/search/shared_tt.rs` (three atomic words per slot checked by XOR, in a 32-byte slot so that none straddles
two cache lines); `History` in `rust/src/search/ordering.rs` (relaxed atomics, a clone is a handle to the same scores).
`Memory::with_threads` (`rust/src/search/memory.rs`) builds the shared table only with more than one thread. UCI options
`Threads` and `Hash` (megabytes of the shared table; with one thread `Hash` has no effect); in the window, Settings →
Advanced. With more than one thread the search is not deterministic.

## Measured effect

`python -m chess_lab.measure.search_speed --threads 1,2,4,6` (60 positions of `data/openings/openings.epd`, `go depth 8`,
`Hash` 256, six-core laptop with a browser open, so the figures are noisy; both columns were measured in the same
session). Seconds to depth 8, mean per position:

| threads | before (helpers at fixed depth, one vote) | now |
|---|---|---|
| 1 | 1.174 | 1.109 |
| 2 | 0.956 | 0.800 |
| 4 | 0.788 | 0.638 |
| 6 | 0.871 | 0.627 |

The earlier measurement on an idle machine (six threads no better than four: 1.211 s against 1.180 s) is in the git
history of this page. Six threads are still no faster than four to depth 8 on this loaded machine.

**In games** (2026-11-04, `data/results/matches/smp_6v4/match.log`, desktop worker with four physical and eight logical cores, `fastchess`, one game at a time, 10 s + 0.1 s, `Hash` 256, SPRT [0, 10]): six threads (A) against four threads (B), `linear_pst_terms` for both:

| test | games | score | Elo (95 %) | pentanomial | LLR | verdict |
|---|---|---|---|---|---|---|
| 6 threads vs 4 threads | 280 scored (317 played) | 50.71 % | +5.0 ± 37.5 | 22, 24, 48, 20, 26 | 0.01 of ±2.94 | stopped by hand, no verdict |

Six threads do not beat four: the result is compatible with zero and the speed figures agree. Why was not isolated; the worker has only four physical cores.

**Vote or no vote at six threads** (2026-11-05, `data/results/matches/smp_vote/match.log`, same setup but `Hash` 256 and a fixed 300 games, no SPRT): the default binary (A) against the same code returning the main thread's move (B, a throwaway build with the vote bypassed; it has no switch), six threads each:

| test | games | score | Elo (95 %) | pentanomial |
|---|---|---|---|---|
| 6 threads with vote vs without, 10 s + 0.1 s | 300 | 44.83 % | -36.0 ± 37.9 (last rating, 280 games) | 37, 23, 44, 26, 20 |

The vote did not help: the point estimate is negative and the interval just reaches zero, so it is not proven to hurt either. It is kept as is until a larger run says otherwise.

## Status

Adopted as an option, off by default (one thread keeps matches reproducible and the clock fair); since 2026-10-25. Four threads are the recommended count to play with (the Lichess bot uses them): six gave no gain over four in 280 games.
Helpers with their own deepening, the shared history, the aligned slots and the vote: 2026-11-04.
