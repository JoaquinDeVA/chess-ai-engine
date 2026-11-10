# Search techniques

> **What it is:** The catalogue of every technique of the search: the core proven by an SPRT, the base, the retired ones and the ideas, with their measurements and pages.
>
> **Read it when:** You want to know what the search does beyond plain alpha-beta, or you are about to add a technique.
>
> **Next:** [Iterative deepening with a clock](iterative-deepening.md)

The plain search is explained in [How the engine chooses a move](../SEARCH.md). Each technique below has one page with the same six parts:
problem, idea, how it works, implementation, measured effect and status ([template](TEMPLATE.md)).

## The core: switches proven by an SPRT

The four switches on by default (`DEFAULT_TECHNIQUES` in `rust/src/search/options.rs`). Each was judged by
removing it from the default set of then, SPRT [−5, 0], in the round robin of 2026-11-06 to 2026-11-09 (fastchess,
10 s + 0.1 s, `Hash` 16, the 10,000 openings of `data/openings/openings.epd`, model `linear_pst_terms`; files
`data/results/roundrobin/abl_<switch>/llr.csv`). A positive figure is what the switch gains.

| Technique | Removal SPRT | Page |
|---|---|---|
| Incremental network accumulator (`incremental`) | +35.5 ± 14.7 Elo, 1,690 games, H1 | [incremental](incremental.md) |
| Table move first (`tt_move`) | +56.5 ± 18.3 Elo, 1,098 games, H1 | [tt-move](tt-move.md) |
| Null move (`null_move`) | +97.4 ± 24.5 Elo, 710 games, H1 | [null-move](null-move.md) |
| Static exchange evaluation (`see`) | +64.5 ± 19.5 Elo, 992 games, H1 | [see](see.md) |

`pvs` and `history` are always on, with no switch (`70701f7`); their SPRTs are running and continue from the earlier LLR:

| Technique | Removal SPRT [−5, 0] | Page |
|---|---|---|
| Principal variation search | LLR +2.21 of ±2.94 so far (`abl_pvs`) | [pvs](pvs.md) |
| History heuristic | LLR +2.09 of ±2.94 so far (`abl_history`) | [history](history.md) |

## Added, SPRT pending

On by default but not yet judged by an SPRT (queued for the compute node).

| Technique | Evidence so far | Page |
|---|---|---|
| Late move reductions (`lmr`) | `ttd 11`: −68 % nodes, 23.53 s to 10.82 s; screening 51.33 % in 300 games | [lmr](lmr.md) |
| Reverse futility pruning (`rfp`) | `ttd 11` with `lmr`: −33.7 % nodes, 11.13 s to 8.58 s median | [rfp](rfp.md) |

**Implemented, off by default**, also waiting for their SPRT:

| Technique | Evidence so far | Page |
|---|---|---|
| Futility pruning (`futility`) | `ttd 11` with `lmr`, `rfp`: −32.8 % nodes, 8.65 s to 6.96 s median | [futility](futility.md) |
| Late move pruning (`lmp`) | same set: −69.3 % nodes, 8.65 s to 3.91 s median | [lmp](lmp.md) |

## The base: always on, no SPRT of its own

These are part of the search itself rather than switches, so they were never removed in a match. Their pages give
what was measured (nodes, speed, positions solved).

| Technique | Status and effect | Page |
|---|---|---|
| Iterative deepening with a clock | on | [iterative-deepening](iterative-deepening.md) |
| Transposition table | on | [transposition-table](transposition-table.md) |
| Killer moves | on | [killer-moves](killer-moves.md) |
| MVV-LVA move ordering | on | [mvv-lva](mvv-lva.md) |
| Quiescence search | on (4 plies) | [quiescence](quiescence.md) |
| Make/unmake, terminal detection | on | [board-state](board-state.md) |
| Opening book | on, in the window | [opening-book](opening-book.md) |
| Endgame tablebases | on | [endgame-tablebases](endgame-tablebases.md) |
| Search in Rust | since 2026-10-25; 304 k to 928 k nodes/s in `uci bench` with `linear_pst_terms` (+130.9 Elo for the 2.58x speed-up of `67f413a`, +180 with the two later ones; 300 games each, 10 s + 0.1 s, [CHANGELOG](../../CHANGELOG.md)); 1,020,805 nodes/s with `net_pst_w32` at `8700e6f` | [rust-core](rust-core.md) |
| Several threads (Lazy SMP) | UCI option `Threads`, 1 by default | [lazy-smp](lazy-smp.md) |

## Retired, to be taken up again

Removed from the code in `efb595e` (2026-11-09) because the round robin did not prove them: none reached a verdict
before it was stopped (`pvs` and `history` came back in `70701f7`). The last version of each is in `a1bdf60`; its page keeps the measurements and says how to
take it up again. Same conditions as the core table; `abl_` jobs remove the switch from the set of then, `add_` jobs
add it.

| Technique | Round-robin SPRT [−5, 0] | Page |
|---|---|---|
| Aspiration windows (`aspiration`) | +10.7 ± 12.5 Elo, 2,200 games, LLR 1.37, no verdict (`abl_aspiration`) | [aspiration](aspiration.md) |
| History malus (`history_malus`) | +9.2 ± 12.0 Elo, 2,200 games, LLR 1.24, no verdict (`add_history_malus`) | [history-malus](history-malus.md) |
| Quiescence with the table (`qs_tt`) | −0.3 ± 12.5 Elo, 2,134 games, LLR 0.19, no verdict (`add_qs_tt`) | [qs-tt](qs-tt.md) |

**Removed after losing.** Three earlier search changes were implemented, measured and deleted; they have no page
(the code is in the git history before the commit that removed them):

| change | measurement | command or file |
|---|---|---|
| Late move reductions (`lmr`, first version, at 20,000 nodes) | −8.6 ± 6.1 Elo, SPRT [−5, 0] at 20,000 nodes, 10,244 games | `data/results/matches/lmr_nodes/match.log` |
| Quiescence evasions (`qs_evasions`) | −21.3 ± 10.6 Elo, SPRT [−5, 0] at 20,000 nodes, 3,428 games | `data/results/matches/qs_evasions/match.log` |
| Clock rules (`TimeRules`: `half`, `stab`, `nodes`; `single` changed no game) | −10.4 ± 30.6, −25.5 ± 33.6, −80.1 ± 37.2 Elo; 300 games at 10 s + 0.1 s each, screening | `data/results/matches/{half,stab,nodes}_vs_base/match.log` |

## Ideas

| Technique | Status |
|---|---|
| Razoring | idea; not implemented |
| Check extensions | idea; not implemented |
| Using the increment in the clock | pending: the UCI binary ignores `winc`/`binc` |

## How the switches were judged

Each switch is a field of `Techniques` (`rust/src/search/options.rs`) and a name in the comma-separated UCI option
`Techniques` of the binary (an empty value turns them all off). The binary and the window (Settings → Advanced)
use the four of the core: `DEFAULT_TECHNIQUES`, which the window reads as `_core.DEFAULT_TECHNIQUES`.

**Before the SPRT instrument** the default set of seven was adopted from one 100-game match per step against the
reference, at equal time (4 s + 0.05 s, colours swapped, `linear_pst_terms`; `scripts/bench/compare_engine_versions.py
--candidate tt_move,history --openings 50 --base 4 --increment 0.05`, removed in `6162db8`; in the git history):

| candidate | score | Elo | mean depth (reference 4.68) |
|---|---|---|---|
| `tt_move,history` | 63.0 % | +92 | 5.20 |
| `+ pvs, aspiration` | 64.5 % | +104 | 5.20 |
| `+ null_move` | 76.5 % (95 % interval 69.0–84.0) | +205 | 5.54 |

Those steps showed that the set gains but did not separate techniques (100 games give intervals of about ±8
points). The round robin judged each switch on its own, and only the ones with an accepted H1 stayed in the core.
A technique added from now on, or a change to one of these, is judged by an SPRT against the current default.

## How to add a technique

1. Copy [TEMPLATE.md](TEMPLATE.md) to `docs/techniques/<name>.md` and fill its six parts.
2. Add a row to the core table (if its SPRT accepts it) or to the retired table (if not), and link the page.
3. Put it behind a switch in `Techniques` (`rust/src/search/options.rs`) and add a test to `rust/tests/techniques.rs`.
4. Measure it with an SPRT match against the current default ([How a change is measured](../MEASUREMENT.md)) and write the figure and its command in the page.
5. If it changes the default, update [RESULTS.md](../RESULTS.md) and the [changelog](../../CHANGELOG.md).
