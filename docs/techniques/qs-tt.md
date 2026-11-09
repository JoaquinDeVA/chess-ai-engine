# Quiescence with the table (`qs_tt`)

> **What it is:** Storing the value of a horizon node's quiescence search in the transposition table; implemented, off by default.
>
> **Read it when:** You wonder whether the quiescence search repeats work, or why this switch is off.
>
> **Next:** [Search techniques](README.md)

## Problem

The [quiescence search](quiescence.md) uses no table. A position reached at the horizon by two move orders, or
again in the next iteration of the [deepening](iterative-deepening.md), has its captures searched again.

## Idea

Keep the result in the [transposition table](transposition-table.md), as any other node does, and reuse it.

## How it works

At a node with no depth left, after its quiescence search, the value is stored with depth 0 and the usual flag
(exact, lower or upper bound, against the window the node came in with) if the table had no entry for the
position. The check at the start of every node, which uses an entry whose depth reaches the one wanted, then
returns it for the next visit.

A quiescence value depends on the plies it had left. Only the node at the horizon stores, and it always starts
with the full 4 plies; the nodes inside the quiescence store nothing. A depth-0 entry is used only by nodes with
no depth left, so it always stands for the full budget. An existing entry is never replaced, and a search cut by
the clock stores nothing.

## Implementation

Removed in `efb595e` (2026-11-09) with its switch, its tests and its UCI name; the last version is in `a1bdf60`. It lived in the horizon branch of `Search::node` in `rust/src/search/node.rs`. Switch: `qs_tt` (not in the default
list), field `qs_tt` of `Techniques` in `rust/src/search/options.rs`. Test:
`the_horizon_table_entries_do_not_change_the_value` in `rust/src/search/tests.rs` (same root value, three
positions).

## Measured effect

**SPRT at a fixed 20,000 nodes per move** (2026-11-02, `data/results/matches/qs_tt/match.log`). The default with `qs_tt`
(A) against the default (B), with the same search effort for both, so the clock plays no part
([How a change is measured](../MEASUREMENT.md)):

```bash
fastchess -engine cmd=rust/target/release/uci name=A option.Techniques=incremental,tt_move,history,pvs,aspiration,null_move,qs_tt \
    -engine cmd=rust/target/release/uci name=B option.Techniques=incremental,tt_move,history,pvs,aspiration,null_move \
    -each proto=uci nodes=20000 option.Hash=16 -openings file=data/openings/openings.epd format=epd order=random \
    -sprt elo0=-5 elo1=0 alpha=0.05 beta=0.05 model=normalized -rounds 5000 -games 2 -concurrency 8 -ratinginterval 100 -srand 21
```

| test | games | score | Elo (95 %) | LLR | pentanomial | verdict |
|---|---|---|---|---|---|---|
| default + `qs_tt` vs default, 20,000 nodes | 10,000 | 50.10 % | +0.7 ± 3.9 | +1.55 | 345, 270, 3,685, 420, 280 | no verdict: round cap reached |

Neither bound was crossed: the interval (−3.2 to +4.6 Elo) rules out a loss of 5 Elo only loosely and shows no
gain. Fixed nodes favour a change that saves nodes ([Measurement](../MEASUREMENT.md)), so a gain here would still
need a test at a time control. The nodes saved and the share of table hits were not recorded.

**SPRT [−5, 0] with the clock**, the default set plus `qs_tt` against the default set. Round robin of 2026-11-06 to 2026-11-09 (fastchess, 10 s + 0.1 s, `Hash` 16, the 10,000 openings of `data/openings/openings.epd`, the model then embedded, `linear_pst_terms`; queue `src/chess_lab/scripts/worker/queue.json` at `d2d03ba`): −0.3 ± 12.5 Elo
at 2,134 games, LLR +0.19 of ±2.94, no verdict when the round robin was stopped
(`data/results/roundrobin/add_qs_tt/llr.csv`).

## Status

Retired, to be taken up again (`efb595e`, 2026-11-09); it was off by default. Neither SPRT gave a verdict, and
neither shows a gain (+0.7 ± 3.9 Elo at 20,000 nodes, −0.3 ± 12.5 at 10 s + 0.1 s). To take it up again: restore it from `a1bdf60` and run its SPRT to a verdict against the current default ([How a change is measured](../MEASUREMENT.md)).
