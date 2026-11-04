# Search techniques

> **What it is:** The catalogue of every technique added to the search: what it is, whether it is on, what it gained, and its page.
>
> **Read it when:** You want to know what the search does beyond plain alpha-beta, or you are about to add a technique.
>
> **Next:** [Iterative deepening with a clock](iterative-deepening.md)

The plain search is explained in [How the engine chooses a move](../SEARCH.md). Each technique below has one page with the same six parts:
problem, idea, how it works, implementation, measured effect and status ([template](TEMPLATE.md)).
Node counts are per move at depth 6 over 100 positions (reference: 672,800); the measurements are in the pages.

| Technique | Status and effect | Elo estimated | Page |
|---|---|---|---|
| Iterative deepening with a clock | reference, on | not measured | [iterative-deepening](iterative-deepening.md) |
| Transposition table | reference, on | not measured | [transposition-table](transposition-table.md) |
| Killer moves | reference, on | not measured | [killer-moves](killer-moves.md) |
| MVV-LVA move ordering | reference, on | not measured | [mvv-lva](mvv-lva.md) |
| Quiescence search | on (4 plies) | not measured | [quiescence](quiescence.md) |
| Make/unmake, terminal detection | reference, on | not measured | [board-state](board-state.md) |
| Opening book | on, in the window | not measured | [opening-book](opening-book.md) |
| Endgame tablebases | on | not measured | [endgame-tablebases](endgame-tablebases.md) |
| Search in Rust | since 2026-10-25 | not measured | [rust-core](rust-core.md) |
| Several threads (Lazy SMP) | UCI option `Threads`, 1 by default | not measured | [lazy-smp](lazy-smp.md) |
| Incremental network accumulator (`incremental`) | on; effect with the light networks not measured | not measured | [incremental](incremental.md) |
| Table move first (`tt_move`) | on; nodes 672,800 to 256,693 | +92 with `history` (100 games, [RESULTS](../RESULTS.md#search-techniques-in-games)) | [tt-move](tt-move.md) |
| History heuristic (`history`) | on; nodes to 391,130 | +92 with `tt_move` (same match) | [history](history.md) |
| Principal variation search (`pvs`) | on; nodes to 487,244 | +104 with the two above and `aspiration` (same match series) | [pvs](pvs.md) |
| Aspiration windows (`aspiration`) | on; nodes to 440,213 | +104 with `pvs` (same) | [aspiration](aspiration.md) |
| Null move (`null_move`) | on; nodes to 446,129 | +205 with the five above (76.5 %, 100 games) | [null-move](null-move.md) |
| Late move reductions (`lmr`) | implemented, off: loses at fixed nodes | −8.6 ± 6.1 (SPRT) | [lmr](lmr.md) |
| Quiescence evasions (`qs_evasions`) | implemented, off: loses at fixed nodes | −21.3 ± 10.6 (SPRT) | [qs-evasions](qs-evasions.md) |
| Quiescence with the table (`qs_tt`) | implemented, off: no gain at fixed nodes | +0.7 ± 3.9 (SPRT, no verdict) | [qs-tt](qs-tt.md) |
| Static exchange evaluation (`see`) | implemented, off: promising, no verdict yet | +47.8 ± 34.3 (screening) | [see](see.md) |
| History malus | idea; not measured | not measured | - |
| Time rules (`TimeRules`: `single`, `half`, `stab`, `nodes`) | implemented, off; `single` not measured | `half` −10.4, `stab` −25.5, `nodes` −80.1 (screening) | [iterative-deepening](iterative-deepening.md) |
| Futility pruning | idea; not measured | not measured | - |
| Razoring | idea; not measured | not measured | - |
| Check extensions | idea; not measured | not measured | - |
| Using the increment in the clock | pending: the UCI binary ignores `winc`/`binc` | not measured | - |

## How the switches were judged

The classic techniques (the rows above `incremental` in the table) and the [Rust core](rust-core.md) form the
*reference* search; the nine switches below them are off in it. Each switch is a field of `Techniques`
(`rust/src/search/options.rs`) and a name in the comma-separated UCI option `Techniques` of the binary (an empty
value turns them all off). The binary and the window (Settings → Advanced) use all of them but `lmr`, `qs_evasions` and `qs_tt`: `DEFAULT_TECHNIQUES` in
`rust/src/search/options.rs`, which the window reads as `_core.DEFAULT_TECHNIQUES`.

**The default set was adopted before the SPRT instrument existed.** These are standard techniques of alpha-beta
engines, so each was checked for what can go wrong (more nodes, a different move: the node counts in its page)
and the set was judged by one 100-game match per step against the reference, at equal time (4 s + 0.05 s,
colours swapped, `linear_pst_terms`; `scripts/bench/compare_engine_versions.py --candidate tt_move,history
--openings 50 --base 4 --increment 0.05`, removed in `6162db8`; in the git history):

| candidate | score | Elo | mean depth (reference 4.68) |
|---|---|---|---|
| `tt_move,history` | 63.0 % | +92 | 5.20 |
| `+ pvs, aspiration` | 64.5 % | +104 | 5.20 |
| `+ null_move` | 76.5 % (95 % interval 69.0–84.0) | +205 | 5.54 |

The steps show that the set gains; they do not separate techniques (100 games give intervals of about ±8 points)
and none has its own SPRT. A technique added from now on, or a change to one of these, is judged by an SPRT
against the current default, as [`lmr`](lmr.md) was.

## How to add a technique

1. Copy [TEMPLATE.md](TEMPLATE.md) to `docs/techniques/<name>.md` and fill its six parts.
2. Add a row to the table above and link the page.
3. Put it behind a switch in `Techniques` (`rust/src/search/options.rs`) and add a test to `rust/tests/techniques.rs`.
4. Measure it with an SPRT match against the current default ([How a change is measured](../MEASUREMENT.md)) and write the figure and its command in the page.
5. If it changes the default, update [RESULTS.md](../RESULTS.md) and the [changelog](../../CHANGELOG.md).
