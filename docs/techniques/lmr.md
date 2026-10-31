# Late move reductions (`lmr`)

> **What it is:** Searching the late quiet moves of an ordered list one ply less; implemented, off by default.
>
> **Read it when:** You wonder why an obvious technique is off, or want to know where its re-test stands.
>
> **Next:** [Tactical move ordering with Prolog](prolog-ordering.md)

## Problem

With good ordering the best move of a node is almost always among the first few. The last quiet moves are rarely
best, but each costs as much as the first.

## Idea

Search late quiet moves with less depth. If the reduced search says one is better than the best so far, it was
not a bad move after all: search it again at full depth.

## How it works

At 4 or more plies left, from the 5th move on, a move is reduced if it is quiet: not a capture, a promotion, a
killer, a move out of check or a move giving check. It is searched one ply less (two from the 15th move with 5 or
more plies left) with a zero-width window at alpha. If that value beats alpha, it goes on to the normal search
(with [PVS](pvs.md), the zero-width search at full depth, then the full window).

**Worked example** (illustrative). A node with 6 plies left and 30 ordered moves. Moves 1–4 and the captures,
killers and checks are searched as usual, at 5 plies. The quiet moves from the 5th to the 14th are searched at 4
plies, those from the 15th at 3. Say the 9th, `a4`, beats alpha at 4 plies: it is searched again at 5, and its full
value counts. The other late quiet moves cost a 3- or 4-ply search instead of a 5-ply one.

## Implementation

`child_value` in `rust/src/search/node.rs`, with `LMR_FROM_MOVE = 5`, `LMR_FROM_DEPTH = 4` and
`LMR_DEEPER_FROM_MOVE = 15`; the quiet test is in the move loop of `Search::node`. Switch: `lmr` (UCI option
`Techniques`, not in the default list).

## Measured effect

Rust core, `linear_pst_terms`, 100 positions at depth 6 (`scripts/bench/compare_search_features.py`, removed in
`6162db8`; in the git history), and 100 games at 4 s + 0.05 s against the reference
(`scripts/bench/compare_engine_versions.py`, same removal):

| version | nodes per move | same move as reference | games |
|---|---|---|---|
| first, aggressive | 86,830 | 70 % | all techniques with it: 63.5 %, below 76.5 % without it |
| mild | 146,356 | 82 % | against the set without it: 49.0 % (−7 Elo), half a ply deeper |

Both searched deeper and played no better.

> [!NOTE]
> **Pending: SPRT at a fixed 20,000 nodes per move.** A re-test of the default with `lmr` (A) against the
> default without it (B) is running ([How a change is measured](../MEASUREMENT.md)): SPRT with bounds [−5, 0] Elo,
> alpha = beta = 0.05, at most 6,000 rounds (12,000 games), 8 games at a time, `Hash` 16 MB. This section will
> get its result:
>
> | test | games | score | Elo (95 %) | LLR | pentanomial | verdict |
> |---|---|---|---|---|---|---|
> | default + `lmr` vs default, 20,000 nodes | pending | pending | pending | pending | pending | pending |
>
> with the fastchess command, the opening suite and the mean depth of each side. A fixed node count compares
> the searches without the clock; if `lmr` does not lose, a match at 10 min + 5 s confirms it with the clock.

## Status

Implemented, off by default; rejected on the evidence of the games above (2026-10-25, `f5e368a`). Pending: the
decision after the SPRT at 20,000 nodes. A stronger evaluation orders moves better and would make reductions
safer, so a negative result now does not close it.
