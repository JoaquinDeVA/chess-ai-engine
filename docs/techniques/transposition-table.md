# Transposition table

> **What it is:** A memory of positions already searched, so the search does not analyse the same position twice.
>
> **Read it when:** You want to know why a deeper iteration costs less than it should, or what the option `Hash` does.
>
> **Next:** [Killer moves](killer-moves.md)

## Problem

The same position is reached through different move orders: 1. e4 e5 2. Nf3 and 1. Nf3 e5 2. e4 lead
to the same board. Iterative deepening makes it worse: the search to depth 6 visits again every
position the search to depth 5 already analysed. A plain alpha-beta search treats each arrival as a
new position and pays for the whole subtree again.

## Idea

Keep a table keyed by the position. After searching a node, write down its value, its best move and
how deep it was searched. When the position comes back, read the entry before searching: if it was
searched at least as deep, its value can be reused, either at once or to narrow the window.

## How it works

A value found inside an `(alpha, beta)` window is not always exact. If a move reaches `beta`, the
node is cut and the value is only a **lower bound** (the real value is at least that); if no move
beats `alpha`, it is only an **upper bound**. So every entry carries a flag computed against the
window the node was entered with:

| value found | flag | meaning |
|---|---|---|
| `≤ alpha` | `UPPERBOUND` | the real value is at most this |
| `≥ beta` | `LOWERBOUND` | the real value is at least this |
| between | `EXACT` | this is the value |

When a position comes back with `depth_remaining` no greater than the stored depth:

- `EXACT`: return the stored value without searching.
- `LOWERBOUND`: raise `alpha` to the stored value.
- `UPPERBOUND`: lower `beta` to the stored value.
- If the window is now closed (`alpha ≥ beta`), return the stored value.

A shallower entry is not used for its value, but its best move is still the best guess for ordering
([table move first](tt-move.md)).

**Worked example.** Values are expected scores for the side the search plays for (`1` win, `0` loss).

1. The node P is entered at depth 4 with the window `(0.40, 0.60)`. Its second move returns `0.65`,
   which reaches `beta = 0.60`: cut. The table stores `P → 0.65, LOWERBOUND, depth 4`.
2. Later, P is reached through another move order, at depth 3, with the window `(0.30, 0.50)`. The
   entry is deep enough (4 ≥ 3). The lower bound raises `alpha` to `max(0.30, 0.65) = 0.65`, which
   is above `beta = 0.50`: the window is closed and P returns `0.65` at once. Its subtree, about
   `30³ ≈ 27,000` nodes without pruning, is not searched.
3. P is reached a third time, at depth 4, with the window `(0.60, 0.80)`. The lower bound raises
   `alpha` to `0.65`; the window `(0.65, 0.80)` is still open, so P is searched, but with a narrower
   window that cuts sooner.
4. At depth 5 the entry (depth 4) is too shallow for its value; only its best move is tried first.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../img/transposition-table-dark.svg">
  <img src="../img/transposition-table-light.svg" alt="Probe of the transposition table: an exact entry returns at once, a lower bound raises alpha, an upper bound lowers beta, and a closed window returns the stored value; a shallow entry only gives its move">
</picture>

## Implementation

`Table` and `Entry` in `rust/src/search/tt.rs`; probed at the start of `Search::node` and written at
its end (`rust/src/search/node.rs`). The key is `GameState::key()` (`rust/src/board.rs`): a hash of
the position and the two move counters. The table is capped at 500,000 entries
(`DEFAULT_MAX_TT_ENTRIES`), oldest dropped first. The UCI option `Hash` (megabytes) sizes only the
shared table of several threads; with one thread it has no effect.
Nothing is stored after the clock truncates a search, because a truncated value is neither exact nor
a bound. The table lives for the whole game (the iterations of a move and the moves that follow);
values are from the point of view of the colour played, so it is cleared on a new game. Quiescence
does not use it. With several threads the table is the lock-free `SharedTable`
([Lazy SMP](lazy-smp.md)). Test: `rust/src/search/tt.rs` (flags against the window).

## Measured effect

Measured with the Python engine, `linear_material` as evaluation, the table empty at the start of every
position (`scripts/bench/compare_search_features.py`, removed in `6162db8`; in the git history):

| sample | nodes per move, with / without | time per move, with / without | same move |
|---|---|---|---|
| 60 positions, depth 4 | 15,988 / 17,184 (−7.0 %) | 1.365 s / 1.474 s | 100 % |
| 20 positions, depth 6 | 224,896 / 185,089 (+21.5 %) | 16.562 s / 14.619 s | 95 % |

At depth 4 the table saves 7 % of the nodes. On the 20 positions at depth 6 the search with it
visited *more* nodes; the sample is small and the cause was not found, so do not read either figure as
a general rule. The saving inside one move is small because each position starts with an empty
table; in a game the table is warm from the previous moves, which this measurement does not cover.

## Status

Adopted; part of the reference search since the first commit (2026-10-16). Size cap and no store after
truncation since `1ee37d6`.
