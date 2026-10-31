# Transposition table

> **What it is:** A memory of positions already searched, so the search does not analyse the same position twice.
>
> **Read it when:** You want to know why a deeper iteration costs less than it should.
>
> **Next:** [Killer moves](killer-moves.md)

**Problem.** The same position can be reached through different move orders, and iterative deepening
repeats the positions of the previous iteration. A plain search analyses them again every time.

**What it does.** Each searched position is stored with its value, best move, depth and a flag
(`EXACT`, `LOWERBOUND`, `UPPERBOUND`) computed against the `(alpha, beta)` window the node was entered
with, since a cut-off value is only a bound. If the position returns with no more depth remaining than
stored, an exact entry is returned at once and a bound tightens the window (returning if it closes it).

**How it is implemented.** `Table` and `Entry` in `rust/src/search/tt.rs`, probed at the start of
`Search::node` and written at its end. A hash map keyed by the position key (`GameState::key()` in
`rust/src/board.rs`), capped at 500,000 entries (`DEFAULT_MAX_TT_ENTRIES`), oldest dropped first;
nothing is stored once the clock has truncated a search. With several threads the table is a
lock-free one shared by all of them ([several threads](lazy-smp.md)). It is shared by the iterations of a move and by the moves of a game, and cleared in
`on_game_start` (values are from the point of view of the colour played). Quiescence does not use it.

**Measured effect.** `python (removed) scripts/bench/compare_search_features.py --player linear_material
--positions 60 --depth 4` (60 positions; the table is empty at the start of every position):

| variant | nodes/move | s/move | same as baseline | Stockfish |
|---|---|---|---|---|
| baseline | 15,988 | 1.365 | 1.000 | 0.317 |
| no transposition table | 17,184 | 1.474 | 1.000 | 0.317 |

At depth 4 the table saves 7.0 % of the nodes (17,184 → 15,988) and 7.4 % of the time, and changes
no move. At depth 6 the sample says the opposite (`--positions 20 --depth 6 --variants
"no transposition table,no killer moves"`):

| variant | nodes/move | s/move | same as baseline | Stockfish |
|---|---|---|---|---|
| baseline | 224,896 | 16.562 | 1.000 | 0.450 |
| no transposition table | 185,089 | 14.619 | 0.950 | 0.400 |

On those 20 positions the baseline visited *more* nodes (+21.5 %) and took more time (+13.3 %)
than the search without the table. The sample is small and the repository does not explain this
result; do not read it as a general rule.

**Status.** Kept; since the first commit (2026-10-16); size cap and no-store-after-truncation since `1ee37d6`.
