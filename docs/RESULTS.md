# Results

> **What it is:** What was measured, what was only estimated, which model plays best, and the command or file behind every number.
>
> **Read it when:** You want to compare models or techniques, check a claim of the other pages, or repeat a measurement.
>

All numbers are from this project, on a six-core CPU. Commands are run from the repository root. Two engines produced them: the Python engine, until
2026-10-25, and the Rust engine since (the changelog has its speed-ups); every section says which
one. Scripts under `scripts/bench/` and `play_tournament` were removed in `6162db8` and are in the git history.

## The short answer

- **The engine that plays is the Rust UCI binary** with its search techniques all fixed (none can be switched
  off). Six won an SPRT against the set without them: incremental accumulator, table move, null move and static
  exchange evaluation (+35.5 to +97.4 Elo), `pvs` and `history` (+22.8 and +20.0; "Round robin" below). `lmr`,
  `rfp`, `iir` and `futility` were added by their node count at a fixed depth; their SPRTs at 60 s + 1 s were stopped
  without a verdict (LLR +0.68 to +1.27, all positive), and they stay on that evidence ("Search techniques at
  60 s + 1 s" below). An earlier `lmr` lost 8.6 ± 6.1 Elo at 20,000 nodes per move (`lmr_nodes` below), so that
  conflict is still open. Four others (`lmp`, `aspiration`, `history_malus`, `qs_tt`) reached no verdict and were
  retired.
- **The model that plays is `net_pst_threats_w64_both`**: piece-square inputs plus five threat inputs, width 64, and a
  second layer that reads the accumulators of both sides over the same weights. It is the default of the binary
  since `4e82cc7`, and the weights are the effort-100 ones since `4c59054`. Against the previous default `net_pst_w64`
  (retired) it scored +62.1 ± 18.5 Elo (H1), and against `net_pst_w64_both` (the same network without the threat
  inputs) +12.1 ± 9.4 in 4,300 games and +40.7 ± 18.5 with the optimised binary ("Network matches" below). Neither
  of those two matches reached an SPRT bound (LLR 1.93 and 2.02 of 2.94); the intervals exclude 0, and the threats are
  kept on that evidence. The effort-100 weights against the effort-30 ones: +10.0 ± 11.5 in 2,722 games, LLR +1.00,
  no verdict and an interval that includes 0.
- **Retired networks.** The width-32 network, the hybrids (classical terms as inputs), the king-bucket networks, the
  three-layer network and the logistic model were retired on 2026-11-15 and their weights deleted. Their figures
  are in "Retired models" below; those measured before 2026-11-13 are orientative (some played without the
  accumulator, against the search of then).

## Measured and estimated

| Claim | How it is known | Engine | Strength of the evidence |
|---|---|---|---|
| The measuring instrument has no bias | Three A/A matches of 400 to 500 games, fastchess ([Measurement](MEASUREMENT.md#the-aa-check)) | Rust | Measured; intervals ±24 to ±27 Elo |
| Each of the four default techniques makes the engine stronger | SPRT [−5, 0] removing it from the default set of then, 710 to 1,690 games at 10 s + 0.1 s ("Round robin" below) | Rust | Measured; H1 accepted for each. The set of four has no SPRT against the set of seven |
| `pvs` and `history` make the engine stronger | Same round robin, 2,424 and 2,778 games | Rust | Measured; H1 accepted (+22.8 ± 11.6, +20.0 ± 11.1) |
| `aspiration`, `history_malus` and `qs_tt` help | Same round robin, 2,134 to 2,200 games each | Rust | Not shown: no verdict (LLR 0.19 to 1.37); retired |
| `lmr`, `rfp`, `iir` and `futility` make the engine stronger | SPRT [−5, 0] at 60 s + 1 s, 400 to 644 games each ("Search techniques at 60 s + 1 s" below) | Rust | Not shown: +34.9 to +62.3 Elo, LLR 0.68 to 1.27, stopped without a verdict; kept in the engine |
| `lmp`, `aspiration`, `history_malus` and `qs_tt` help on top of `lmr,rfp,iir` | Same SPRTs, 200 to 600 games each | Rust | Not shown: LLR −0.48 to 0.07, stopped; retired |
| The network plays better than the logistic model (retired) | SPRT [0, 5] of 504 games at 10 s + 0.1 s, +165.1 ± 32.9 Elo | Rust | Orientative (measured before 2026-11-13); H1 accepted |
| Adding the classical terms to the network makes it weaker (retired) | SPRT [0, 5] of 672 games at 10 s + 0.1 s, −105.5 ± 25.5 Elo | Rust | Orientative: H0 accepted, but it mixes quality and speed (the hybrids had no accumulator) |
| The threat inputs make the network stronger | `sp_threats_both_vs_both_opt`, SPRT [0, 5], 1,200 games at 10 s + 0.2 s, +40.7 ± 18.5 Elo | Rust | Measured; LLR 2.02, no bound reached, interval excludes 0 (the same match with the binary before the speed-ups: 4,300 games, +12.1 ± 9.4, LLR 1.93). `sp_threats_both_vs_w64` against the older network reached H1 (+62.1 ± 18.5) |
| The effort-100 weights play better than the effort-30 ones | `sp_threats_e100_vs_e30`, SPRT [0, 5], 2,722 games at 10 s + 0.2 s, +10.0 ± 11.5 Elo | Rust | Not shown: LLR +1.00, interval includes 0; the weights that play, with the lower error of their card |
| Late move reductions made the engine weaker (removed) | SPRT of 10,244 games at 20,000 nodes per move: 48.76 %, −8.6 ± 6.1 Elo | Rust | Measured at 20,000 nodes; H0 accepted. Applying it only in null-window nodes later reached H1 at 10 s + 0.2 s ("Search technique matches") |
| The integer search of `345c3ff` is faster than 4.0.0 | `uci ttd 11 bench/middlegame.epd 60` (one thread, desktop, alternated with the build before the change): median 5.06 s to 4.19 s, 919,477 to 1,268,761 nodes/s; `uci bench 5` 0.127 s to 0.105 s | Rust | Measured speed; the tree changed (branching factor 2.12 to 2.38), so strength was judged by `rust_cleanup_vs_v4`, SPRT [−5, 0] against `v4.0.0`, 896 games at 10 s + 0.1 s: +73.6 ± 21.0 Elo, LLR +2.97, H1 accepted |
| Several threads search faster | Time to depth 8 on 60 positions (Lazy SMP) | Rust | Measured speed; effect in games not measured |
| Training on quiet positions makes the model stronger | SPRT of 362 games at 60 s + 1 s, +284 ± 43 Elo (logistic model) | Rust | Measured; H1 accepted |
| Quiescence, endgame tables and the opening book help | Position suites ("Decisions that are settled") | Python | Measured on positions, not in games |

## Round robin (Rust engine, fastchess)

From 2026-11-06 to 2026-11-09 the compute node played every job of `src/chess_lab/measure/queue.json` at
`d2d03ba` in turns (`src/chess_lab/measure/roundrobin.py`): 10 s + 0.1 s, `Hash` 16, one thread, the 10,000
openings of `data/openings/openings.epd`, LLR bounds ±2.94. Technique jobs used the default set of then
(`incremental,tt_move,history,pvs,aspiration,null_move,see`) and `linear_pst_terms`; `abl_` removes one switch from
it (A is the full set, so a positive Elo is what the switch gains), `add_` adds one. Model jobs used the default
techniques. It was stopped on 2026-11-09 with jobs still running; their figures are the last report. Measured on the compute node with `src/chess_lab/measure/roundrobin.py`; the per-job `llr.csv` files were removed from the node on 2026-11-19, so the figures are the last state of each job.

| job | A against B | SPRT | games | Elo (95 %) | LLR | verdict |
|---|---|---|---|---|---|---|
| `abl_null_move` | default against default without `null_move` | [−5, 0] | 710 | +97.4 ± 24.5 | +2.95 | H1: keep |
| `abl_see` | default against default without `see` | [−5, 0] | 992 | +64.5 ± 19.5 | +2.96 | H1: keep |
| `abl_tt_move` | default against default without `tt_move` | [−5, 0] | 1,098 | +56.5 ± 18.3 | +2.96 | H1: keep |
| `abl_incremental` | default against default without `incremental` | [−5, 0] | 1,690 | +35.5 ± 14.7 | +2.96 | H1: keep |
| `abl_pvs` | default against default without `pvs` | [−5, 0] | 2,424 | +22.8 ± 11.6 | +2.98 | H1: keep |
| `abl_history` | default against default without `history` | [−5, 0] | 2,778 | +20.0 ± 11.1 | +2.97 | H1: keep |
| `abl_aspiration` | default against default without `aspiration` | [−5, 0] | 2,200 | +10.7 ± 12.5 | +1.37 | no verdict; retired |
| `add_history_malus` | default + `history_malus` against default | [−5, 0] | 2,200 | +9.2 ± 12.0 | +1.24 | no verdict; retired |
| `add_qs_tt` | default + `qs_tt` against default | [−5, 0] | 2,134 | −0.3 ± 12.5 | +0.19 | no verdict; retired |

The model jobs of that round robin are in "Retired models" below.

## Search techniques at 60 s + 1 s

From 2026-11-11 the compute node played one SPRT [−5, 0] per technique at 60 s + 1 s in a new round robin, with
the binary `db98489bd531` (the last one with switches) on both sides, `Hash` 16, one thread. `abl_` removes one
technique from the default set of then, `lmr,rfp,iir` (A is the full set, so a positive Elo is what the technique
gains); `add_` adds one to it. On 2026-11-13 the four `add_` jobs that did not move were stopped and their techniques
retired; the other four were stopped as well, without a verdict, while the engine already had them fixed. Measured on the compute node with `src/chess_lab/measure/roundrobin.py`; the per-job `llr.csv` files were removed from the node on 2026-11-19, so the figures are the last state of each job.

| job | A against B | games | Elo (95 %) | LLR | state |
|---|---|---|---|---|---|
| `abl_lmr_60s` | `lmr,rfp,iir` against `rfp,iir` | 644 | +42.6 ± 25.5 | +1.25 | stopped; `lmr` fixed |
| `abl_rfp_60s` | `lmr,rfp,iir` against `lmr,iir` | 400 | +62.3 ± 31.4 | +1.13 | stopped; `rfp` fixed |
| `abl_iir_60s` | `lmr,rfp,iir` against `lmr,rfp` | 400 | +34.9 ± 30.8 | +0.68 | stopped; `iir` fixed |
| `add_futility_60s` | `lmr,rfp,iir,futility` against `lmr,rfp,iir` | 400 | +44.5 ± 32.0 | +0.82 | stopped; `futility` fixed |
| `add_lmp_60s` | `lmr,rfp,iir,lmp` against `lmr,rfp,iir` | 600 | −19.1 ± 24.5 | −0.48 | stopped; retired |
| `add_aspiration_60s` | `lmr,rfp,iir,aspiration` against `lmr,rfp,iir` | 600 | −1.7 ± 23.9 | +0.01 | stopped; retired |
| `add_history_malus_60s` | `lmr,rfp,iir,history_malus` against `lmr,rfp,iir` | 236 | +4.4 ± 39.2 | +0.07 | stopped; retired |
| `add_qs_tt_60s` | `lmr,rfp,iir,qs_tt` against `lmr,rfp,iir` | 200 | −8.7 ± 43.7 | −0.06 | stopped; retired |

The engine with the eight fixed techniques has no SPRT against the one with switches; `uci bench 5` counts 23.5 %
fewer nodes (139,252 to 106,583).

## Network matches

From 2026-11-14 the compute node played the networks against each other with the binary `242475e64e00` (the one of
`both` and `threats` before the speed-ups): 10 s + 0.2 s, `Hash` 16, one thread, SPRT [0, 5], LLR bounds ±2.94. A
is the first network. Measured on the compute node with `src/chess_lab/measure/roundrobin.py`; the per-job `llr.csv` files were removed from the node on 2026-11-19, so the figures are the last state of each job.

| job | A against B | games | Elo (95 %) | LLR | state |
|---|---|---|---|---|---|
| `sp_both_vs_w64` | `net_pst_w64_both` against the retired `net_pst_w64` | 1,486 | +45.1 ± 15.7 | +2.95 | H1 (2026-11-16) |
| `sp_threats_both_vs_w64` | `net_pst_threats_w64_both` against the retired `net_pst_w64` | 1,092 | +62.1 ± 18.5 | +2.97 | H1 (2026-11-16) |
| `sp_threats_both_vs_both` | `net_pst_threats_w64_both` against `net_pst_w64_both` | 4,300 | +12.1 ± 9.4 | +1.93 | no verdict (stopped 2026-11-19); the next row repeats it with the binary after the speed-ups of the threat inputs |
| `sp_threats_both_vs_both_opt` | the same, with the binary `5d862f1d7451` (`2272d51`) | 1,200 | +40.7 ± 18.5 | +2.02 | stopped by hand; accepted as valid |
| `sp_threats_e100_vs_e30` | `net_pst_threats_w64_both` trained at effort 100 against the same network at effort 30, binary `242475e64e00` | 2,722 | +10.0 ± 11.5 | +1.00 | no verdict (stopped 2026-11-19); interval includes 0; the effort-100 weights play since `4c59054` |
| `sp_threats_both_vs_w64_nodes` | `net_pst_threats_w64_both` against the retired `net_pst_w64` at 250,000 nodes per move | 652 | +112.5 ± 25.7 | +2.96 | H1: stronger per node |
| `sprt_mpk_w128_e15_vs_pst_w128_e15` | `net_ctx_mpk_w128_both@e15` (material, pawn files and king buckets) against `net_pst_w128_both@e15`, binary `9f3075cebf16` | 628 | −104.9 ± 23.7 | −2.96 | H0 (2026-11-22): its lower validation loss (0.62928 against 0.63002) did not carry into play |
| `sprt_mpk_w128_e100_vs_mpk_w128_e15` | `net_ctx_mpk_w128_both` at effort 100 against effort 15, binary `9f3075cebf16` | 4,600 | +6.2 ± 8.7 | +0.89 | no verdict (stopped 2026-11-23 after the H0 above) |
| `sprt_pst_w256_e15_vs_pst_w128_e15` | `net_pst_w256_both@e15` against `net_pst_w128_both@e15`, binary `9f3075cebf16` | 4,800 | +6.6 ± 8.3 | +1.06 | no verdict (stopped 2026-11-23); interval includes 0 |
| `nodes_ctx_mpk_w128_e15_vs_pst_w128_e15` | the first `ctx` row at 135,000 nodes per move | 200 | −75.9 ± 42.9 | −0.72 | worse per node (stopped 2026-11-23: the interval excludes 0) |
| `nodes_ctx_m_w128_e15_vs_pst_w128_e15` | `net_ctx_m_w128_both@e15` (material only) against `net_pst_w128_both@e15` at 135,000 nodes per move | 176 | −86.4 ± 53.7 | −0.58 | worse per node (stopped 2026-11-23: the interval excludes 0) |
| `nodes_ctx_p_w128_e15_vs_pst_w128_e15` | `net_ctx_p_w128_both@e15` (pawn files only) against `net_pst_w128_both@e15` at 135,000 nodes per move | 1,000 | −30.7 ± 18.0 | −1.61 | worse per node (stopped 2026-11-23: the interval excludes 0) |
| `nodes_ctx_k_w128_e15_vs_pst_w128_e15` | `net_ctx_k_w128_both@e15` (king buckets only) against `net_pst_w128_both@e15` at 135,000 nodes per move | 920 | −53.7 ± 20.0 | −2.36 | worse per node (stopped 2026-11-23: the interval excludes 0) |
| `nodes_ob8_w128_e15_vs_pst_w128_e15` | `net_pst_w128_ob8_both@e15` (eight output buckets by piece count, binary `b425eef37187`) against `net_pst_w128_both@e15` at 135,000 nodes per move | 600 | −15.7 ± 23.7 | −0.52 | no verdict (stopped 2026-11-23 before its 2,000-game cap); interval includes 0 |

From 2026-11-23 a match stopped by hand counts as a verdict when its 95 % interval excludes 0, and as no verdict otherwise.
All four `ctx` families lose per node (and `mpk` at 10 s + 0.2 s as well), so the `ctx` inputs lose on quality, not on the cost of the evaluation.
The `ctx` rows (2026-11-22 on) keep their `llr.csv` in `data/results/roundrobin/<job>/`. The row with the optimised binary has its own file because the binary is the one that plays; the speed-up of the threats
(`2272d51`, `8402069`) is in the changelog. `net_pst_w64_both` reaches depth 11 in 20 % less time than `net_pst_w64`
(`uci ttd 11`), and `net_pst_threats_w64_both` costs 13 % more time than `net_pst_w64_both` over the 20 positions (`uci ttd 11 bench/middlegame.epd 60`, laptop, 2026-11-23: 9.0 s against 8.0 s, 1,559,605 against 1,787,601 nodes/s; the median, 0.42 s against 0.32 s, mixes different trees). Computing the threats twice in `uci bench 20` drops the nodes per second from about 1,555 k to 1,400 k: about 72 ns of the 96 ns a node costs more are the threats themselves.

## Search technique matches

From 2026-11-14 the compute node played each search change against the build before it: 10 s + 0.2 s, `Hash` 16, one
thread, `net_pst_w64`, SPRT [0, 5], LLR bounds ±2.94. A is the build with the change, built from the commit in the
changelog. Measured on the compute node with `src/chess_lab/measure/roundrobin.py`; the per-job `llr.csv` files were removed from the node on 2026-11-19, so the figures are the last state of each job.

| job | A against B | games | Elo (95 %) | LLR | state |
|---|---|---|---|---|---|
| `lmr_null_window` | late move reductions in null-window nodes (`953501b`) against without | 970 | +73.1 ± 20.6 | +2.95 | H1 (2026-11-14) |
| `check_extension` | extension by one ply in check (part of `6f23616`) against its parent `0fbd5a5` | 2,376 | +29.2 ± 12.5 | +2.96 | H1 (2026-11-15) |
| `qs_evasions` | check evasions at the horizon (`6f23616`) against the extension alone | 2,616 | +27.2 ± 12.1 | +2.97 | H1 (2026-11-16) |
| `killers_by_ply` | killers by ply (`6bf589a`) against `953501b` | 5,900 | +9.3 ± 7.9 | +1.94 | no verdict (stopped 2026-11-19); kept |
| `null_r_by_eval` | null-move reduction by depth and margin (`63c4b0d`) against `6f23616` | 4,300 | +1.5 ± 9.1 | −0.15 | no verdict (stopped 2026-11-19); reverted in `ec990da` |
| `cont_history` | continuation history (`0f1ac06`) against the build with countermove | 3,400 | +3.0 ± 10.4 | +0.12 | no verdict (stopped 2026-11-19); kept |
| `history_gravity` | history gravity (`2055295`) against `0f1ac06` | 1,336 | +50.5 ± 16.6 | +2.95 | H1 (2026-11-18) |
| `futility_margin_90` | futility margin 90 (`940b6d9`) against 120 | 4,010 | +12.9 ± 9.2 | +2.09 | no verdict (stopped 2026-11-19); kept |
| `legal_after_prune` | legal moves generated after reverse futility and the null move (`82b4f28`) against `7104df4`, with `net_pst_threats_w64_both@e100` | 4,664 | +13.8 ± 7.7 | +2.96 | H1 (2026-11-23) |

`legal_after_prune` (2026-11-22 on) keeps its `llr.csv` in `data/results/roundrobin/legal_after_prune/`.

The binaries of `cont_history`, `history_gravity` and `futility_margin_90` were built on top of countermove ordering and
a reverse futility margin of 60, both retired in `c3ff94c` after their own SPRTs drifted negative (`countermove`
−2.9 ± 17.6, LLR −0.28, 1,200 games; `rfp_margin_60` −16.0 ± 14.1, LLR −1.64, 2,000 games; both stopped). The engine
that plays today is therefore not exactly the one these three matches measured. The same retirement stopped `hash_256`
(`Hash` 256 against 16; −7.2 ± 16.3, LLR −0.56, 1,200 games): it is a run-time option and there was nothing to revert.

## Retired models

Retired on 2026-11-15; the weights were deleted (the code is in the history before `23696c0`). All matches: SPRT [0, 5],
10 s + 0.1 s, binary `cc258520bbdb` unless said (the node's configurations name `62998381c19b` for the `acc_` and `pst_` jobs and the unhashed `./uci` for the `model_` ones, which cannot be checked now). Measured on the compute node with `src/chess_lab/measure/roundrobin.py`; the per-job `llr.csv` files were removed from the node on 2026-11-19, so the figures are the last state of each job. **Orientative:**
they are earlier than 2026-11-13 and some played without the accumulator (the hybrids and the king buckets until
2026-11-09) or with the search of then.

| Model | What it looked at | Match | Games | Elo (95 %) | LLR |
|---|---|---|---|---|---|
| `linear_pst_terms` (logistic) | piece squares and classical terms, no hidden layer | `model_net_vs_linear`, `net_pst_w32` against it | 504 | +165.1 ± 32.9 | +2.95 |
| `net_pst_w32` | piece squares, width 32 | `pst_w64_vs_w32`, `net_pst_w64` against it | 1,028 | +73.4 ± 21.5 | +2.95 |
| `net_hybrid_*` | piece squares plus 11 classical terms | `acc_hybrid_w64_vs_pst_w32`, `net_hybrid_w64` against `net_pst_w32` | 2,666 | +26.5 ± 11.9 | +2.95 |
| | | `acc_hybrid_k4_vs_w64`, `net_hybrid_k4_w64` against `net_hybrid_w64` | 5,010 | −9.4 ± 8.8 | −2.65 |
| `net_pst_k4_w64`, `net_pst_k4_threats_w64_both` | piece squares by the rival king's bucket (4 buckets) | `pst_k4_w64_vs_w32` / `pst_k4_w64_vs_w64` | 1,600 / 988 | +26.8 ± 16.0 / −31.9 ± 21.5 | +1.72 / −1.39 |
| `net_pst_w64x16` | a third layer of 16 units | `pst_w64x16_vs_w32` / `pst_w64x16_vs_w64` | 1,970 / 1,600 | +23.5 ± 14.4 / −38.8 ± 15.9 | +1.84 / −2.91 |
| `net_hybrid_w64` | width 64 against width 32 of the hybrid | `model_width_64_vs_32`, `net_hybrid_w64` against `net_hybrid_w32` | 1,612 | +44.0 ± 15.9 | +2.95 |
| | the hybrid at 250,000 nodes per move | `model_terms_in_net_nodes`, `net_hybrid_w64` against `net_pst_w32` | 784 | +89.2 ± 22.5 | +2.95 |
| `net_hybrid_w64`, `net_hybrid_k4_w64` | against the logistic model | `best_net_hybrid_w64_vs_linear` / `best_net_hybrid_k4_w64_vs_linear` against `linear_pst_terms@e30` | 888 / 900 | +82.1 ± 22.2 / +78.9 ± 21.6 | +2.95 / +2.95 |

Rows with two matches give the network against `net_pst_w32` and against `net_pst_w64`; an LLR short of ±2.94 is a
stopped match. Rule for retiring a model: for quality, when against the current one with the same binary and time
control its interval excludes a gain above +5 Elo and it ties or loses; for cost, when the nodes per second it
loses outweigh the quality it brings. A cost retirement can be reopened if the cost falls.

## Engine matches (Rust engine, fastchess)

Earlier matches of the UCI binary against itself, one change at a time, as explained in
[How a change is measured](MEASUREMENT.md). Logs in `data/results/matches/<name>/match.log`.

| match | A against B | conditions | games | score | Elo (95 %) | LLR | verdict |
|---|---|---|---|---|---|---|---|
| `aa_s11` | default against default | 10 s + 0.1 s, SPRT [−5, 0] | 400 counted | 50.5 % | +3.5 ± 26.8 | +0.11 | no bias |
| `aa_s12` | default against default | 10 s + 0.1 s, SPRT [−5, 0] | 500 | 51.3 % | +9.0 ± 25.0 | +0.28 | no bias |
| `aa_s13` | default against default | 10 s + 0.1 s, SPRT [−5, 0] | 500 | 50.7 % | +4.9 ± 23.9 | +0.18 | no bias |
| `lmr_nodes` | default + `lmr` against default | 20,000 nodes per move, SPRT [−5, 0] | 10,244 | 48.76 % | −8.6 ± 6.1 | −2.94 | H0: regresses |
| `qs_evasions` | default + `qs_evasions` against default | 20,000 nodes per move, SPRT [−5, 0] | 3,428 | 46.94 % | −21.3 ± 10.6 | −2.96 | H0: regresses (the switch of 2026-11-04, retired; the evasions in check at the horizon of `6f23616` reached H1 at 10 s + 0.2 s, "Search technique matches") |
| `see_nodes` | default + `see` against default | 20,000 nodes per move, SPRT [0, 5] | 1,188 | 58.04 % | +56.4 ± 17.5 | +2.97 | H1: stronger (fixed nodes) |
| `history_malus_nodes` | default + `history_malus` against default | 20,000 nodes per move, SPRT [0, 5] | 2,404 | 50.56 % | +3.9 ± 9.3 | +0.33 | no verdict (stopped by hand at 2,556 games; Elo as of the last report, 2,404 games) |
| `qs_tt` | default + `qs_tt` against default | 20,000 nodes per move, SPRT [−5, 0] | 10,000 | 50.10 % | +0.7 ± 3.9 | +1.55 | no verdict (round cap) |
| `quiet_vs_actual` | `linear_pst_terms` trained on quiet positions against the same trained on all positions | 60 s + 1 s, SPRT [0, 5] | 362 | 83.70 % | +284 ± 43 | +2.96 | H1: stronger |
| `q_k65_vs_quiet` | quiet model with `k` 0.0065 against the quiet model | 60 s + 1 s, SPRT [0, 5] | 280 | 48.93 % | −7 ± 35 | −0.13 | no verdict (round cap) |
| `q_m05_vs_quiet` | quiet model with 0.5 % of the mates against the quiet model | 60 s + 1 s, SPRT [0, 5] | 280 | 43.75 % | −44 ± 36 | −0.59 | no verdict (round cap) |
| `q30_vs_quiet` | quiet model with 30 M positions against the quiet model | 60 s + 1 s, SPRT [0, 5] | 240 | 53.33 % | +23 ± 41 | +0.22 | no verdict (stopped by hand) |
| `reloj_nuevo` | new clock against the old clock, same model and techniques | 60 s + 1 s, SPRT [0, 5] | 720 | 40.83 % | −64 ± 24 | −2.06 | no verdict (stopped by hand); the new clock is worse |
| `tm_vs_legacy` | new time manager (soft and hard limit, increment, scaling by best-move stability and node share) against the old one (`1e3b203`), same model and techniques | 60 s + 1 s, SPRT [0, 5], `Hash` 16, one thread | 1,020 | 54.56 % | +31.8 ± 17.3 | +1.53 | no verdict (stopped by hand at 1,030 games; figures as of the last report, 1,020 games; 441-348-231, 0 losses on time) |
| `see_vs_base` | default + `see` against default | 10 s + 0.1 s, 300 games (screening) | 300 | 56.83 % | +47.8 ± 34.3 | - | screening only |
| `see_sprt` | default + `see` against default | 60 s + 1 s, SPRT [0, 5] | 160 | 55.94 % | +41.5 ± 46.1 | +0.30 | no verdict (stopped by hand) |
| `see_sprt10` | default + `see` against default | 60 s + 1 s, SPRT [0, 10] | 282 | 57.27 % | +50.9 ± 38.6 | +1.13 | no verdict (stopped by hand at 303 games; Elo as of the last report, 282 games) |
| `passed_vs_base` | `linear_pst_terms` plus passed-pawn features against `linear_pst_terms` | 10 s + 0.1 s, 300 games (screening) | 300 | 54.17 % | +29.0 ± 35.2 | - | screening only |
| `king_vs_base` | `linear_pst_terms` plus passed-pawn and king-attack features against `linear_pst_terms` | 10 s + 0.1 s, 300 games (screening) | 300 | 50.00 % | 0.0 ± 35.3 | - | screening only |
| `e30_vs_base` | `linear_pst_terms` with 30 M positions (`linear_pst_terms@e30`) against `linear_pst_terms` | 10 s + 0.1 s, 300 games (screening) | 300 | 55.67 % | +39.6 ± 33.8 | - | screening only |

## How the models are measured

| Judge | What it measures | Command |
|---|---|---|
| Static | Spearman correlation and error of the model against Stockfish on 200,000 test positions (seed 42), scored by the Rust core as the engine plays it (clipped to 0.01–0.99); µs per eval is the median of one core evaluation from a FEN over 1,000 positions | `python -m chess_lab.measure.evaluate --model net_pst_threats_w64_both@e100` |
| Match | Games of the UCI binary against itself with fastchess, Elo with a 95 % interval | [Measurement](MEASUREMENT.md) |

## Search techniques in games

The Rust engine reaches 2 to 3 plies more than the Python one in the same time (the changelog has its speed-ups). The six techniques on by default before
`see` (search techniques) scored 76.5 % against the plain search with the same model in 100
games at 4 s + 0.05 s, reaching 5.54 plies against 4.68 (`scripts/bench/compare_engine_versions.py --candidate
incremental,tt_move,history,pvs,aspiration,null_move --openings 50 --base 4 --increment 0.05`, removed in `6162db8`).

## Decisions that are settled

| Question | Answer | Where it comes from |
|---|---|---|
| Keep every mate position in training? | No: 10 % of them. `linear_pst_terms` trained that way scored 71.0 % against 29.0 % for the all-mates version (100 games, effort 15, 60 s + 1 s, Python engine). Keeping 0.5 % instead did not differ beyond the error (−44 ± 36 Elo, `q_m05_vs_quiet`). | `DEFAULT_MATE_KEEP = 0.10`; `python -m chess_lab.training.train --model linear_pst_terms --effort 15 --mate-keep 1.0 --tag mates100` (at `a1bdf60`, before the logistic model was retired) and a match against the default model |
| Train on quiet positions only? | Yes: +284 ± 43 Elo against the all-positions model (362 games, SPRT [0, 5] accepted). Three variants (`k` 0.0065, 0.5 % of the mates, 30 M positions) did not differ from it beyond the error (−7 ± 35, −44 ± 36, +23 ± 41 Elo). | `data/results/matches/quiet_vs_actual/match.log`, `q_k65_vs_quiet`, `q_m05_vs_quiet`, `q30_vs_quiet`; Models |
| Is quiescence worth its cost? | Yes: at depth 3 it adds +0.047 (`h1`) and +0.096 (`linear_pst_terms`) to the share of positions where the move is Stockfish's (standard error 0.015), at 2.5 to 3.6 times the nodes. | `scripts/bench/compare_quiescence.py --players h1,linear_pst_terms` (removed in `6162db8`); Quiescence |
| Why did games stop at depth 2? | The original rule that predicts the next iteration (last duration squared over the previous one) over-estimated it. It was replaced by `geometric_growth`, with a floor of depth 3. | `scripts/bench/compare_time_estimators.py` (removed in `6162db8`); Iterative deepening |
| Do the endgame tables help? | Yes: from 60 won endgames of 3 to 5 pieces the search with the tables converts 52 and without them 8. | `scripts/bench/compare_endgame_tablebase.py --positions 60 --seconds 0.5` (removed in `6162db8`); Endgame tablebases |
| Is the opening book good? | 98.7 % of 2,438 sampled moves are within 50 centipawns of Stockfish's best line, none illegal. | `python src/chess_lab/datasets/check_opening_book.py data/opening_book.json --fraction 0.02 --engine /usr/games/stockfish --depth 16` |

## Limits and leakage

- **Positions, not games.** The data has no game identifiers, so the split is by a hash of the position.
  `python src/chess_lab/datasets/check_split_leakage.py` found no duplicate position among the 124,923,027
  rows; 1.1 % of the test positions have their colour-swapped image in the data and 72.0 % have a
  position one move away (60.2 % in train).
- **The neighbours do not flatter the models.** Judged separately on test positions with and without a neighbour
  in train, the Spearman correlation is not higher with it for `linear_pst_terms` (table in Data).
- **Statistical power.** Read every match with its interval; the screenings of 300 games have about ±35 Elo.
- **One machine, one run.** The figures depend on the CPU (the 1 s depth in particular).

## Model table

Generated by `python -m chess_lab.measure.report` from the model cards; do not edit by hand. It holds what can
be measured again today (accuracy and cost); the model cards carried depth and search-agreement figures until 2026-11-01.

<!-- BEGIN GENERATED TABLES -->
| Model | Params | Representation | Effort | Spearman | MAE | Excess (nats) | Best move, all / non-quiet | µs per eval |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `net_ctx_k_w128_both` | 393,601 | winprob-k0.00368-stm-quiet | 15 | 0.741 | 0.106 | 0.0760 | 21.6% / 56.9% | 2.2 |
| `net_ctx_m_w128_both` | 102,529 | winprob-k0.00368-stm-quiet | 15 | 0.740 | 0.105 | 0.0748 | 22.1% / 57.7% | 2.6 |
| `net_ctx_mpk_w128_both` | 401,537 | winprob-k0.00368-stm-quiet | 15 | 0.744 | 0.104 | 0.0741 | 21.9% / 57.7% | 3.2 |
| `net_ctx_p_w128_both` | 102,785 | winprob-k0.00368-stm-quiet | 15 | 0.737 | 0.107 | 0.0765 | 21.9% / 57.7% | 2.6 |
| `net_pst_threats_w64_both` | 49,665 | winprob-k0.00368-stm-quiet | 100 | 0.751 | 0.103 | — | 24.7% / 60.0% | 2.1 |
| `net_pst_w128_both` | 98,689 | winprob-k0.00368-stm-quiet | 15 | 0.744 | 0.105 | 0.0748 | 22.4% / 57.8% | 2.1 |
| `net_pst_w128_ob8_both` | 100,488 | winprob-k0.00368-stm-quiet | 15 | 0.739 | 0.106 | 0.0755 | 22.2% / 58.5% | 4.5 |
| `net_pst_w256_both` | 197,377 | winprob-k0.00368-stm-quiet | 15 | 0.747 | 0.104 | 0.0742 | 22.6% / 58.0% | 3.0 |
| `net_pst_w64_both` | 49,345 | winprob-k0.00368-stm-quiet | 30 | 0.731 | 0.110 | — | 22.7% / 58.3% | 2.4 |
| `h1` | 6 | — | — | 0.561 | 0.149 | — | 19.6% / 49.8% | 2.8 |
<!-- END GENERATED TABLES -->

## Reproduce

The pipeline rebuilds the sidecars, trains the models, evaluates them and regenerates the model table; it plays no
games (games are played with fastchess, [Measurement](MEASUREMENT.md)).

```bash
DRY_RUN=1 bash src/chess_lab/pipeline.sh   # prints every command
nohup setsid bash src/chess_lab/pipeline.sh > data/results/pipeline/pipeline.log 2>&1 &
```
