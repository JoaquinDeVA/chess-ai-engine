# Static exchange evaluation (`see`)

> **What it is:** A count of what a capture wins or loses once both sides have recaptured on the square, used to skip losing captures; implemented, off by default.
>
> **Read it when:** You wonder how the search tells a free pawn from a queen given for a defended pawn, or why this switch is off.
>
> **Next:** [Search techniques](README.md)

## Problem

[MVV-LVA](mvv-lva.md) orders captures by the victim and the capturer, and the [quiescence search](quiescence.md) tries every capture. A queen taking a defended pawn looks the same as a queen taking an undefended one, so the search spends nodes on exchanges that lose material.

## Idea

Play out the exchange on the target square without searching: each side recaptures with its cheapest attacker and stops when going on loses. The result is the material the capture wins or loses.

## How it works

`see_ge(position, move, threshold)` answers whether the exchange started by the move gives at least `threshold`. It swaps attackers on the square, cheapest first (values P 1, N 3, B 3, R 5, Q 9, K 100), including the attackers uncovered behind a piece that has captured (a rook battery), and lets either side stop. With `see` on:

- The quiescence search skips a capture with `see_ge(…, 0)` false, except when the side is in check and `qs_evasions` is on (without it the check gets no special treatment and SEE prunes there too). The test ignores pins.
- The main search orders captures that fail the test after the quiet moves.

## Implementation

`see_value` and `see_ge` in `rust/src/search/ordering.rs`; their use in `Search::node` (`rust/src/search/node.rs`). Switch: `see` (UCI option `Techniques`, not in the default list), field `see` of `Techniques` in `rust/src/search/options.rs`. Tests: `see_tells_winning_from_losing_captures` (`ordering.rs`) and `see_keeps_the_obvious_capture` (`rust/src/search/tests.rs`).

## Measured effect

**Screening, 300 games at 10 s + 0.1 s** (2026-11-04, `data/matches/see_vs_base/match.log`, command `~/worker/cribar.sh`, 7 games at a time): the default with `see` (A) against the same binary without it (B), `linear_pst_terms` for both:

```bash
fastchess -engine cmd=./uci_base name=A option.Techniques=incremental,tt_move,history,pvs,aspiration,null_move,see \
    -engine cmd=./uci_base name=B -each proto=uci tc=10+0.1 option.Hash=16 \
    -openings file=openings.epd format=epd order=random -rounds 150 -games 2 -repeat -concurrency 7 -srand 21
```

| test | games | score | Elo (95 %) | pentanomial |
|---|---|---|---|---|
| default + `see` vs default, 10 s + 0.1 s | 300 | 56.83 % | +47.8 ± 34.3 | 17, 12, 65, 25, 31 |

A screening only decides whether a longer test is worth it; the interval is wide. An SPRT [0, 5] at 60 s + 1 s, stopped by hand after 160 games, gave +41.5 ± 46.1 Elo and an LLR of 0.30 of ±2.94 (no verdict): [0, 5] separates two close values and needs about a thousand games to settle an effect this size. An SPRT [0, 10] (same conditions), stopped by hand after 303 games, gave +50.9 ± 38.6 Elo (as of 282 games) and an LLR of 1.13 of ±2.94 (no verdict). Nodes saved and node counts at depth 6 were not recorded.

## Status

Implemented, off by default. Screening passed (+47.8 ± 34.3 Elo); SPRT [0, 5] at 60 s + 1 s without verdict (+41.5 ± 46.1 in 160 games); SPRT [0, 10] without verdict (+50.9 ± 38.6, LLR 1.13). Consistent positive sign in three tests; not enough to switch it on.
