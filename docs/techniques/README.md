# Search techniques

> **What it is:** The catalogue of every technique added to the search: what it is, whether it is on, what it gained, and its page.
>
> **Read it when:** You want to know what the search does beyond plain alpha-beta, or you are about to add a technique.
>
> **Next:** [Iterative deepening with a clock](iterative-deepening.md)

The plain search is explained in [How the engine chooses a move](../SEARCH.md). Each technique below has one page with the same six parts:
problem, idea, how it works, implementation, measured effect and status ([template](TEMPLATE.md)).
Node counts are per move at depth 6 over 100 positions (reference: 672,800); the measurements are in the pages.

| Technique | Status and effect | Page |
|---|---|---|
| Iterative deepening with a clock | reference, on | [iterative-deepening](iterative-deepening.md) |
| Transposition table | reference, on | [transposition-table](transposition-table.md) |
| Killer moves | reference, on | [killer-moves](killer-moves.md) |
| MVV-LVA move ordering | reference, on | [mvv-lva](mvv-lva.md) |
| Quiescence search | on (4 plies) | [quiescence](quiescence.md) |
| Make/unmake, terminal detection | reference, on | [board-state](board-state.md) |
| Opening book | on, in the window | [opening-book](opening-book.md) |
| Endgame tablebases | on | [endgame-tablebases](endgame-tablebases.md) |
| Search in Rust | since 2026-10-25; 8 to 35 times faster | [rust-core](rust-core.md) |
| Several threads (Lazy SMP) | UCI option `Threads`, 1 by default | [lazy-smp](lazy-smp.md) |
| Incremental network accumulator (`incremental`) | on; +15 % nodes/s for `net4_pst` | [incremental](incremental.md) |
| Table move first (`tt_move`) | on; nodes 672,800 to 256,693 | [tt-move](tt-move.md) |
| History heuristic (`history`) | on; nodes to 391,130 | [history](history.md) |
| Principal variation search (`pvs`) | on; nodes to 487,244 | [pvs](pvs.md) |
| Aspiration windows (`aspiration`) | on; nodes to 440,213 | [aspiration](aspiration.md) |
| Null move (`null_move`) | on; nodes to 446,129; +205 Elo with the set | [null-move](null-move.md) |
| Late move reductions (`lmr`) | implemented, off: lost games; SPRT at fixed nodes pending | [lmr](lmr.md) |
| Tactical ordering with Prolog | removed | [prolog-ordering](prolog-ordering.md) |
| History malus | idea; not measured | - |
| Static exchange evaluation (SEE) | idea; not measured | - |
| Futility pruning | idea; not measured | - |
| Razoring | idea; not measured | - |
| Check extensions | idea; not measured | - |
| Using the increment in the clock | pending: the UCI binary ignores `winc`/`binc` | - |

## How the switches were judged

The classic techniques (the rows above `incremental` in the table) and the [Rust core](rust-core.md) form the
*reference* search; the seven switches below them are off in it. Each switch is a field of `Techniques`
(`rust/src/search/options.rs`) and a name in the comma-separated UCI option `Techniques` of the binary (an empty
value turns them all off). The binary and the window (Settings → Advanced, `DEFAULT_TECHNIQUES` in
`src/chess_engine/gui/settings.py`) use all of them but `lmr`. The measurements below were taken with
scripts removed in `6162db8` (in the git history), on the Rust core through its Python module.

**How they were judged.** Two measurements, both repeatable.

1. Nodes and time at depth 6 over 100 positions, and how often the move equals the reference's:
   `scripts/bench/compare_search_features.py` (variants `with <technique>` and `with all of them`),
   Position-terms model:

| variant | nodes per move | s per move | same move as reference |
|---|---|---|---|
| reference | 672,800 | 1.86 | - |
| `tt_move` | 256,693 | 0.71 | 98 % |
| `history` | 391,130 | 1.16 | 100 % |
| `pvs` | 487,244 | 1.33 | 100 % |
| `aspiration` | 440,213 | 1.20 | 100 % |
| `null_move` | 446,129 | 1.25 | 99 % |
| `lmr` (first, aggressive) | 86,830 | 0.23 | 70 % |
| `lmr` (mild) | 146,356 | 0.395 | 82 % |

2. Games against the reference at equal time (4 s + 0.05 s per move, 100 games, colours swapped, same model):
   `scripts/bench/compare_engine_versions.py --model linear_pst_terms --candidate tt_move,history --openings 50 --base 4 --increment 0.05`.

| candidate | score | Elo | mean depth (reference 4.68) |
|---|---|---|---|
| `tt_move,history` | 63.0 % | +92 | 5.20 |
| `+ pvs, aspiration` | 64.5 % | +104 | 5.20 |
| `+ null_move` | 76.5 % (95 % interval 69.0-84.0) | +205 | 5.54 |
| all, aggressive `lmr` | 63.5 % | +96 | 6.26 |
| mild `lmr` against the set without it | 49.0 % | -7 | 6.08 against 5.55 |

A technique is on if it does not lose to the reference and gains depth or time. Fewer nodes alone is not
enough: `lmr` searched deeper and played worse. The intervals of 100 games are wide (about +-8 points), so
the table shows what was decided, not small differences.

## How to add a technique

1. Copy [TEMPLATE.md](TEMPLATE.md) to `docs/techniques/<name>.md` and fill its six parts.
2. Add a row to the table above and link the page.
3. Put it behind a switch in `Techniques` (`rust/src/search/options.rs`) and add a test to `rust/tests/techniques.rs`.
4. Measure it with an SPRT match against the current default ([How a change is measured](../MEASUREMENT.md)) and write the figure and its command in the page.
5. If it changes the default, update [RESULTS.md](../RESULTS.md) and the [changelog](../../CHANGELOG.md).
