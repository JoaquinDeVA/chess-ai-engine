# Changelog

Every behaviour change (`feat`, `fix`, `perf`, retirements) is one row: date, commit, change, the metric measured and its
value before → after with the percentage change. Refactors, tests and documentation are counted at the end of each
milestone. Milestones are dated; no git tags exist. The newest are first.

How the numbers were measured (all single thread unless stated; `docs/MEASUREMENT.md` has the procedure):

- **ttd 9 / ttd 11**: `uci ttd <depth> rust/bench/middlegame.epd 30` — nodes and time to fixed depth over the 20 middlegame
  positions. Nodes are deterministic with one thread, so the change in nodes is exact; time is the median of alternated runs.
- **bench**: `uci bench` (the `bench 5`/`bench 6` variants search to that depth) — nodes per second, median of alternated runs.
- **Elo SPRT**: fastchess matches with a sequential probability ratio test; the bounds, games and time control are in the
  metric; the full table is in `docs/RESULTS.md`. «No verdict» means the test was stopped before it reached a bound.
- Rows measured on 2026-11-14 compare the parent commit with the commit, 6 alternated runs of each (3 for `7fd40b5` and
  `ac56301`, whose tree has no pruning and takes two minutes per run), on a laptop; differences under about 2 % in nodes
  per second are within the noise of that machine.
- «estimated» marks a figure taken from a single run; «not measured» means no benchmark or match exists for the change.


## Unreleased

| Date | Commit | Type | Change | Metric | Before → after | Δ |
|---|---|---|---|---|---|---|
| 2026-11-16 | `d75439c` | feat | each hyper-parameter trial uses 10 % of the positions but at most 5 M rows (effort 100 on ~190 M rows: 5 M instead of 19 M) | — | not measured | — |
| 2026-11-16 | `1d79fe1` | feat | effort is a percentage of the available TRAIN rows again; the epoch limit uses effort + 10 (effort 100: 12 epochs) and patience goes from 3 (effort 0) to 5 (effort 100) | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-11-16 | `bd7adcf` | feat | raw conversion of the whole Lichess evaluation database with `--min-depth 20` and all cores: 417,862,375 lines → 391,000,241 positions (before 124,923,027) | `python -m chess_lab.datasets.build_raw_shards --input data/LichessPositions/lichess_db_eval.jsonl.zst --min-depth 20` (6,075 s, 68,784 lines/s) | 124.9 M → 391.0 M positions | +213 % |
| 2026-11-16 | `c3ff94c` | revert | retire countermove ordering and put the reverse futility margin back to 80 per ply (both were still running their SPRT and drifting negative; hash 256 is only a run-time option, nothing to revert) | bench 5 total nodes<br>Elo SPRT [0,5] 10+0.2, binaries 2fdb9a9cdc81 vs 7c27ecf5ed6c, 800 games (countermove)<br>Elo SPRT [0,5] 10+0.2, binaries c2bc273f7f34 vs 679bf1852d8a, 1,600 games (rfp 60) | 65,462 → 75,659<br>−14.3 ± 21.5 (LLR −0.62, no verdict)<br>−26.5 ± 15.6 (LLR −2.08, no verdict) | +15.6 %<br>—<br>— |
| 2026-11-16 | `3081a48` | fix | search the root when a legal move would claim threefold repetition (it returned the first legal move without searching, losing a queen in game 0X5XjbEr) | bench 5 total nodes<br>`uci bench 5` on the game position, nodes | 65,462 → 65,462<br>30 → 735,816 (depth 30 → 13, best move e8e5) | +0.0 %<br>— |
| 2026-11-15 | `57bf14d` | retire | delete the weights and cards of the retired models (width 32, hybrid, king buckets, three layers) | Elo SPRT [0,5] net_pst_k4_w64 vs net_pst_w64, 988 games 10+0.1 (orientative)<br>Elo SPRT [0,5] net_pst_w64x16 vs net_pst_w64, 1,600 games 10+0.1 (orientative) | -31.9 ± 21.5 (LLR -1.39)<br>-38.8 ± 15.9 (LLR -2.91) | —<br>— |
| 2026-11-15 | `a01a938` | retire | retire the hybrid, king-bucket and classical-term feature sets | Elo SPRT [0,5] net_hybrid_w64 vs net_pst_w32, 2,666 games 10+0.1 (before 2026-11-13: orientative) | +26.5 ± 11.9 (H1); vs net_pst_w64 no verdict | — |
| 2026-11-15 | `4e82cc7` | feat | play net_pst_threats_w64_both by default and let the bot follow the binary's model | Elo SPRT [0,5] threats_both vs net_pst_w64, 800 games 10+0.2 (Network matches)<br>Elo SPRT [0,5] threats_both vs net_pst_w64_both, binary 5d862f1d7451, 1,200 games 10+0.2<br>nodes, bench 5 (the default model changes, so the fingerprint changes) | +57.9 ± 22.1 (LLR +1.98, no verdict)<br>+40.7 ± 18.5 (LLR +2.02, no verdict; accepted as valid)<br>77,589 → 65,462 | —<br>—<br>−15.6 % |
| 2026-11-15 | `8402069` | perf | take the fast output path also when the net has dense inputs | nodes, ttd 11<br>nodes/s, ttd 11 (threats_both, median) | 11,589,046 → 11,589,046<br>909,826 → 978,991 | +0.0 %<br>+7.6 % |
| 2026-11-15 | `2272d51` | perf | compute the threat inputs from one attack map per side | nodes, bench 5<br>nodes/s, bench (median) | 65,462 → 65,462<br>925,000 → 1,000,000 | +0.0 %<br>+8.1 % |
| 2026-11-15 | `7d4712c` | feat | let uci ttd measure any model file | — | not measured | — |
| 2026-11-14 | `61bc5aa` | fix | use as_chunks in the clipped dot product (clippy chunks_exact_to_as_chunks) | nodes, ttd 9<br>nodes/s, ttd 9 (median)<br>nodes/s, bench (median) | 3,989,480 → 3,989,480<br>1,135,065 → 1,136,709<br>1,327,826 → 1,328,686 | +0.0 %<br>+0.1 %<br>+0.1 % |
| 2026-11-14 | `d5ead4d` | feat | threat inputs (pst_threats, pst_k4_threats) and both perspectives with king buckets | — | not measured | — |
| 2026-11-14 | `40bb83e` | perf | cache the pawn-structure terms by pawn placement | nodes, ttd 9<br>nodes/s, ttd 9 (median)<br>nodes/s, bench (median) | 3,989,480 → 3,989,480<br>1,127,892 → 1,129,553<br>1,321,936 → 1,322,626 | +0.0 %<br>+0.1 %<br>+0.1 % |
| 2026-11-14 | `9a1431d` | perf | one-pass accumulator update, fused first-layer dot and a single check test per futility move | nodes, ttd 9<br>nodes/s, ttd 9 (median)<br>nodes/s, bench (median) | 3,989,480 → 3,989,480<br>1,059,501 → 1,131,966<br>1,230,055 → 1,319,300 | +0.0 %<br>+6.8 %<br>+7.3 % |
| 2026-11-14 | `e2e1d9c` | feat | model card of net_pst_w64_both@e30 | — | — | — |
| 2026-11-14 | `b9f62a3` | perf | read and pad training frames in a thread ahead of the training step | — | — | — |
| 2026-11-14 | `0c1f92d` | feat | net_pst_w64_both, a network that reads both sides' accumulators over shared weights | — | not measured | — |
| 2026-11-14 | `4658873` | feat | per-position move-ordering metric against Stockfish | — | — | — |
| 2026-11-14 | `24835ae` | feat | UCI pondering with go ponder, ponderhit and a search_job module | — | — | — |
| 2026-11-14 | `20fa44d` | feat | search limits that a running search can change, with a budget start for the soft scale | — | — | — |
| 2026-11-14 | `52a0e2b` | perf | gate detailed search counters behind the search-diagnostics feature | nodes/s, ttd 9 (12 alternated runs) | 1,010,980 → 1,007,256 | -0.4 % |
| 2026-11-14 | `2055295` | feat | history gravity and penalty for the quiet moves that did not cut | nodes, ttd 9<br>nodes/s, ttd 9 (median)<br>nodes/s, bench (median) | 5,527,170 → 3,989,480<br>1,130,127 → 1,051,756<br>1,306,930 → 1,216,308 | -27.8 %<br>-6.9 %<br>-6.9 % |
| 2026-11-14 | `0f1ac06` | feat | continuation history in the quiet move order | nodes, ttd 9<br>nodes/s, ttd 9 (median)<br>nodes/s, bench (median) | 5,808,947 → 5,527,170<br>1,155,329 → 1,131,296<br>1,321,429 → 1,309,442 | -4.9 %<br>-2.1 %<br>-0.9 % |
| 2026-11-14 | `99d0ba5` | feat | countermove ordering | nodes, ttd 9<br>nodes/s, ttd 9 (median)<br>nodes/s, bench (median) | 6,071,291 → 5,808,947<br>1,187,962 → 1,154,336<br>1,344,424 → 1,320,930 | -4.3 %<br>-2.8 %<br>-1.7 % |
| 2026-11-14 | `940b6d9` | feat | futility margin 90 per ply | nodes, ttd 9 | 6,450,756 → 6,071,291 | -5.9 % |
| 2026-11-14 | `bfd3f44` | feat | reverse futility margin 60 per ply | nodes, ttd 9 | 7,099,976 → 6,450,756 | -9.1 % |
| 2026-11-14 | `63c4b0d` | feat | null-move reduction by depth and static margin | nodes, ttd 9 | 7,416,608 → 7,099,976 | -4.3 % |
| 2026-11-14 | `6f23616` | feat | search check evasions at the horizon | nodes, ttd 9<br>nodes/s, ttd 9 (median)<br>nodes/s, bench (median) | 7,251,582 → 7,416,608<br>1,229,193 → 1,199,128<br>1,350,301 → 1,329,456 | +2.3 %<br>-2.4 %<br>-1.5 % |
| 2026-11-14 | `0fbd5a5` | fix | do not cut a restricted root with the transposition table | — | not measured | — |
| 2026-11-14 | `2f8b646` | fix | keep only the fastest winning root moves from the endgame tables | — | — | — |
| 2026-11-14 | `8831543` | feat | restrict the root moves to those the endgame tables keep winning | self-play wins vs draws, mate without repetition | 100/100 (3 seeds, 20 ms/move) | — |
| 2026-11-14 | `8036a4e` | fix | rank winning zeroing moves first in the endgame tables | — | — | — |
| 2026-11-13 | `d49298a` | retire | fixed search techniques; retire lmp, aspiration, history_malus and qs_tt | nodes, bench 5 (lmr, rfp, iir and futility fixed)<br>LLR SPRT 60+1 of lmp / aspiration / history_malus / qs_tt | 139,252 → 106,583<br>-0.48 / 0.01 / 0.07 / -0.06 of ±2.94 (600, 600, 236, 200 games) | -23.5 %<br>— |
| 2026-11-13 | `6bf589a` | fix | index killer moves by search ply | nodes, ttd 9<br>nodes/s, ttd 9 (median)<br>nodes/s, bench (median) | 7,829,138 → 7,251,582<br>1,217,183 → 1,225,274<br>1,334,105 → 1,344,128 | -7.4 %<br>+0.7 %<br>+0.8 % |
| 2026-11-13 | `953501b` | feat | apply late move reductions in null-window nodes | nodes, ttd 9<br>nodes/s, ttd 9 (median)<br>nodes/s, bench (median) | 14,784,110 → 7,829,138<br>1,172,827 → 1,214,925<br>1,302,008 → 1,334,076 | -47.0 %<br>+3.6 %<br>+2.5 % |
| 2026-11-13 | `b7176bf` | feat | clock horizon 60 minus the move, iteration cost 3.5, skip iterations ending 4x past the soft limit | — | not measured | — |
| 2026-11-13 | `dd4bf59` | feat | clock horizon 50 to 20 by move, soft share 0.4, skip iterations the hard limit would cut | seconds left on the clock at move 30 (300+3 game) | 49 → 173 | +253.1 % |
| 2026-11-13 | `ae9135e` | other | Mejorada legibilidad bench.rs | — | — | — |
| 2026-11-13 | `f9b6231` | fix | dot product with as_chunks for the new clippy lint | — | — | — |

Internal, not itemised: 10 docs, 10 refactor, 1 test, 1 ci (`4f859e3`…`23696c0`).

## 4.0.0 (2026-11-11)

| Date | Commit | Type | Change | Metric | Before → after | Δ |
|---|---|---|---|---|---|---|
| 2026-11-11 | `2e63224` | feat | soft time limit scaled by best-move stability and node share | Elo SPRT [0,5] 60+1 vs the previous clock, 1,020 games | +31.8 ± 17.3 (54.56 %, LLR +1.53 of ±2.94, no verdict) | — |
| 2026-11-11 | `7ea1849` | feat | clock limits from the increment, movestogo and a move overhead | Elo SPRT [0,5] 60+1 vs the previous clock, 1,020 games | +31.8 ± 17.3 (54.56 %, LLR +1.53 of ±2.94, no verdict) | — |
| 2026-11-11 | `1e3b203` | feat | book match suite and an opening file per round robin job | — | — | — |
| 2026-11-11 | `edccccd` | feat | deep book at depth 16 by default (depth 20 with four lines is four times slower) | — | — | — |
| 2026-11-11 | `44e85c9` | feat | opening book built with a UCI engine, most likely positions first | — | — | — |
| 2026-11-11 | `fa602b8` | feat | measure where and why the engine leaves the opening book in the bot's games | — | — | — |
| 2026-11-11 | `53fcd11` | feat | queue the SPRT of the eight techniques with a switch | — | — | — |
| 2026-11-11 | `af88217` | feat | aspiration, history_malus and qs_tt come back as switches on the integer search | — | not measured | — |
| 2026-11-11 | `c3f650b` | feat | round robin jobs can give engine B its own binary (uci_b) | — | — | — |
| 2026-11-11 | `345c3ff` | perf | integer scores, f32 network with accumulator stack and a single lock-free table | median s, ttd 11<br>nodes/s, ttd 11<br>nodes, bench 5 (the tree changes) | 5.06 → 4.19<br>919,477 → 1,268,761<br>145,163 → 139,252 | -17.2 %<br>+38.0 %<br>-4.1 % |
| 2026-11-11 | `8a8370a` | feat | script that saves offscreen screenshots of the window | — | — | — |
| 2026-11-11 | `818b315` | fix | CI mirror-symmetry test compares indices and values apart; actions on Node 24 | — | — | — |

Internal, not itemised: 9 docs, 4 refactor, 3 chore (`d0fb6f5`…`c5029ab`).

## 3.7.0 (2026-11-10)

| Date | Commit | Type | Change | Metric | Before → after | Δ |
|---|---|---|---|---|---|---|
| 2026-11-10 | `052f390` | fix | Lichess bot at four threads, the measured count | — | — | — |
| 2026-11-10 | `17b8dec` | feat | Lichess bot on the compute node with a supervisor that resumes the round robin | — | — | — |
| 2026-11-10 | `4d8dfc4` | feat | net_pst_w64 as the default model of the UCI binary | — | not measured | — |
| 2026-11-10 | `e3c99c4` | feat | internal iterative reduction, on by default | nodes, ttd 11<br>median s, ttd 11<br>Elo SPRT [-5,0] 60+1 without iir, 400 games | 215,120,644 → 108,502,704<br>8.22 → 4.15<br>+34.9 ± 30.8 (LLR +0.68, no verdict) | -49.6 %<br>-49.5 %<br>— |
| 2026-11-10 | `9a783d9` | feat | futility and late move pruning as switches, off by default until their SPRT | nodes, ttd 11 (futility)<br>nodes, ttd 11 (lmp)<br>Elo SPRT 60+1 with futility, 400 games | 215,120,644 → 144,499,072<br>215,120,644 → 66,027,302<br>+44.5 ± 32.0 (LLR +0.82, no verdict) | -32.8 %<br>-69.3 %<br>— |
| 2026-11-10 | `d51b3c8` | feat | reverse futility pruning, on by default | nodes, ttd 11 (with lmr)<br>median s, ttd 11<br>Elo SPRT [-5,0] 60+1 without rfp, 400 games | 324,595,891 → 215,120,644<br>11.13 → 8.58<br>+62.3 ± 31.4 (LLR +1.13, no verdict) | -33.7 %<br>-22.9 %<br>— |
| 2026-11-10 | `fe09c6c` | feat | late move reductions, on by default, with the TTD_TECHNIQUES switch of the bench | nodes, ttd 11<br>median s, ttd 11<br>Elo SPRT [-5,0] 60+1 without lmr, 600 games | 1,004,922,392 → 324,595,891<br>23.53 → 10.82<br>+46.0 ± 26.2 (LLR +1.27, no verdict) | -67.7 %<br>-54.0 %<br>— |
| 2026-11-10 | `ed609a1` | perf | try the captures of the quiescence search by gain | nodes, ttd 9 | 113,050,947 → 107,372,423 | -5.0 % |
| 2026-11-10 | `ac56301` | perf | raise the single-thread table to 2,000,000 entries | nodes, ttd 9<br>nodes/s, ttd 9 (median)<br>nodes/s, bench (median) | 117,358,262 → 113,050,947<br>1,411,930 → 1,355,338<br>1,469,831 → 1,445,233 | -3.7 %<br>-4.0 %<br>-1.7 % |
| 2026-11-10 | `a0c25dd` | feat | key the transposition table by position alone, with mate distance and fifty-move guards | nodes, ttd 9 | 122,806,437 → 117,358,262 | -4.4 % |
| 2026-11-10 | `7fd40b5` | feat | scale the mate preference to 0.0001 per ply | nodes, ttd 9<br>nodes/s, ttd 9 (median)<br>nodes/s, bench (median) | 122,806,437 → 122,806,437<br>1,298,650 → 1,273,714<br>1,371,155 → 1,339,314 | +0.0 %<br>-1.9 %<br>-2.3 % |
| 2026-11-10 | `97e209f` | perf | update the Zobrist key from the move instead of hashing the position again | nodes/s, bench (3 commits together) | 1,088,861 → 1,316,298 | +20.9 % |
| 2026-11-10 | `b9a5021` | perf | order the moves on the stack with a stable in-place sort, no Vec per node | nodes/s, bench (3 commits together) | 1,088,861 → 1,316,298 | +20.9 % |
| 2026-11-10 | `98a471f` | perf | read the clock every 2,048 nodes instead of at every node | nodes/s, bench (3 commits together) | 1,088,861 → 1,316,298 | +20.9 % |
| 2026-11-10 | `252be26` | feat | add uci ttd, the time-to-depth bench, and search counters | nodes, bench (same tree) | 349,598 → 349,598 | +0.0 % |
| 2026-11-10 | `0bc7b69` | feat | run the round robin with 6 concurrent games | — | — | — |
| 2026-11-10 | `59a33dd` | feat | add the GPU-trained net_pst_w64, net_pst_w64x16 and net_pst_k4_w64 and re-time every model | — | — | — |
| 2026-11-10 | `d2c029b` | feat | let entrenar.sh resume only the models in SOLO | — | — | — |

Internal, not itemised: 5 docs, 1 chore (`95c7ed0`…`ceed09e`).

## 3.6.0 (2026-11-09)

| Date | Commit | Type | Change | Metric | Before → after | Δ |
|---|---|---|---|---|---|---|
| 2026-11-09 | `efb595e` | retire | retire pvs, aspiration, history, history_malus and qs_tt switches | Elo SPRT [-5,0] without aspiration, 2,200 games 10+0.1<br>Elo SPRT [-5,0] with history_malus, 2,200 games 10+0.1<br>Elo SPRT [-5,0] with qs_tt, 2,134 games 10+0.1 | +10.7 ± 12.5 (no verdict: retired)<br>+9.2 ± 12.0 (no verdict: retired)<br>-0.3 ± 12.5 (no verdict: retired) | —<br>—<br>— |
| 2026-11-09 | `8700e6f` | feat | the UCI binary plays the piece-square network net_pst_w32 instead of linear_pst_terms | Elo SPRT [0,5] net_pst_w32 vs linear_pst_terms@e30, 504 games 10+0.1<br>nodes/s, bench 5, laptop | +165.1 ± 32.9 (H1)<br>1,020,805 | —<br>— |
| 2026-11-09 | `6d15f3a` | feat | chain the node work after training and queue the 10+5 match alone | — | — | — |
| 2026-11-09 | `70701f7` | feat | make pvs and history part of the core; aspiration, history_malus and qs_tt stay retired | Elo SPRT [-5,0] without pvs, 2,200 games 10+0.1<br>Elo SPRT [-5,0] without history, 2,200 games 10+0.1 | +18.2 ± 12.1 (LLR +2.21, no verdict)<br>+17.5 ± 12.4 (LLR +2.09, no verdict) | —<br>— |
| 2026-11-09 | `3719da0` | revert | restore pvs, aspiration, history, history_malus and qs_tt behind their switches | — | not measured | — |
| 2026-11-09 | `9aade78` | feat | make the uci binary play the opening book and the endgame tablebases | — | — | — |
| 2026-11-09 | `91f807f` | feat | add the entrenar task that sets up and trains the candidate networks on the node | — | — | — |
| 2026-11-09 | `a7b19e4` | feat | add the pst_k4 king-bucket feature set and net_pst_k4_w64 | — | not measured | — |
| 2026-11-09 | `75a8071` | feat | incremental accumulator for bucketed and hybrid networks | nodes/s, bench 5, net_hybrid_w64 | 521,915 → 635,828 | +21.8 % |
| 2026-11-09 | `a1bdf60` | feat | install each engine build in its own path and queue the last linear measurements | — | — | — |

Internal, not itemised: 5 docs, 2 refactor (`107b9e7`…`d7ca746`).

## 3.5.0 (2026-11-07)

| Date | Commit | Type | Change | Metric | Before → after | Δ |
|---|---|---|---|---|---|---|
| 2026-11-07 | `3147519` | feat | add a stop task for the round robin on the compute node | — | — | — |
| 2026-11-06 | `d2d03ba` | feat | also play the best networks directly against the linear model | — | — | — |
| 2026-11-06 | `88c31c3` | feat | compare the models in steps instead of all against the linear one | — | — | — |
| 2026-11-06 | `3d8c467` | feat | start the round robin from the compute node and tolerate a missing log when collecting | — | — | — |
| 2026-11-06 | `35387f0` | feat | delegate the uci upload and the SPRT collection to the compute node | — | — | — |
| 2026-11-06 | `b531f9a` | feat | load networks and linear models from a JSON file in the uci binary | — | — | — |
| 2026-11-06 | `4654821` | feat | add a round-robin scheduler for SPRT matches on the worker | — | — | — |

Internal, not itemised: 2 docs (`63427fb`…`20c6fb7`).

## 3.4.0 (2026-11-05)

| Date | Commit | Type | Change | Metric | Before → after | Δ |
|---|---|---|---|---|---|---|
| 2026-11-05 | `696a14e` | feat | add the trained net_hybrid_w64 and linear_pst_terms@e30 models | Spearman, hybrid networks vs the logistic model<br>Elo SPRT [0,5] hybrid vs piece network, 672 games 10+0.1 | 0.646 → 0.702-0.722<br>-105.5 ± 25.5 (H0) | —<br>— |
| 2026-11-05 | `65eca04` | feat | add hybrid network features with classical terms and king buckets | — | — | — |

Internal, not itemised: 3 docs, 1 refactor (`8c79749`…`ec58665`).

## 3.3.0 (2026-11-04)

| Date | Commit | Type | Change | Metric | Before → after | Δ |
|---|---|---|---|---|---|---|
| 2026-11-04 | `274a911` | retire | remove the rejected lmr, qs_evasions and TimeRules switches | Elo SPRT [-5,0] 20,000 nodes with lmr, 10,244 games<br>Elo SPRT [-5,0] 20,000 nodes with qs_evasions, 3,428 games | -8.6 ± 6.1 (H0: retired)<br>-21.3 ± 10.6 (H0: retired) | —<br>— |
| 2026-11-04 | `ae126c0` | perf | follow the piece-square part of linear_pst_terms move by move, 17 % more nodes per second | nodes/s, bench<br>Elo vs el binary previo al 2.6x (without see), 280 games 10+0.1 | 788,000 → 928,000<br>+180.2 ± 36.0 (73.83 %) | +17.8 %<br>— |
| 2026-11-04 | `1f6b2c7` | perf | build for x86-64-v3 (AVX2, BMI2), same nodes, about 5 % more nodes per second | nodes/s, bench | 750,000 → 788,000 | +5.1 % |
| 2026-11-04 | `cd31e87` | feat | helper threads with their own deepening and a vote, shared history, cache-aligned table slots | s to depth 8, 4 threads<br>s to depth 8, 6 threads<br>Elo 6 threads vs 4, 280 games 10+0.1 | 0.788 → 0.638<br>0.871 → 0.627<br>+5.0 ± 37.5 (no verdict) | -19.0 %<br>-28.0 %<br>— |
| 2026-11-04 | `57a23aa` | feat | add the pst_kings feature set and the net_kings_w32 model | — | — | — |
| 2026-11-04 | `80628d5` | feat | add the history_malus switch, off by default after an SPRT with no verdict | — | not measured | — |
| 2026-11-04 | `5dfeb8d` | feat | switch static exchange evaluation on by default after its SPRT at fixed nodes | Elo SPRT [0,5] 20,000 nodes, 1,188 games<br>Elo SPRT [-5,0] without see, 992 games 10+0.1 | +56.4 ± 17.5 (H1)<br>+64.5 ± 19.5 (H1: keep) | —<br>— |
| 2026-11-04 | `360e5c2` | fix | apply --patience to the training plan that is printed and saved in the card | — | — | — |
| 2026-11-04 | `33497b7` | feat | stop training after N epochs without improvement (--patience) | — | — | — |
| 2026-11-04 | `67f413a` | perf | speed up the Rust search 2.6x with identical nodes (packed move-order key, bitboard pawns, no per-node allocations) | nodes/s, bench<br>Elo vs el binary anterior, 300 games 10+0.1 | 304,000 → 784,000<br>+130.9 ± 39.8 | +157.9 %<br>— |

Internal, not itemised: 10 docs, 4 refactor (`78de930`…`9b0d00b`).

## 3.2.0 (2026-11-03)

| Date | Commit | Type | Change | Metric | Before → after | Δ |
|---|---|---|---|---|---|---|
| 2026-11-03 | `30cba50` | feat | restore the share-of-time clock and add the TimeRules option (single, half, stab, nodes) | — | not measured | — |
| 2026-11-03 | `6caf4dc` | feat | add the see technique (static exchange evaluation, off by default) | — | no change (off by default) | — |
| 2026-11-03 | `2e848cb` | feat | add passed-pawn and king-attack term models (linear_pst_passed, linear_pst_king) | — | — | — |
| 2026-11-03 | `7cd6e18` | feat | ship the model trained on quiet positions as linear_pst_terms | Elo SPRT [0,5] 60+1 vs the model trained on all positions, 362 games | +284 ± 43 (H1) | — |
| 2026-11-02 | `542e4e0` | feat | spend the time left over the moves to go plus 75 % of the increment | Elo SPRT 60+1 vs the previous clock, 720 games | -64 ± 24 (no verdict, the new one is worse) | — |
| 2026-11-02 | `e2cea5b` | feat | add the qs_evasions and qs_tt quiescence switches, off by default | — | not measured | — |

Internal, not itemised: 3 docs (`1865170`…`ab3ca2d`).

## 3.1.0 (2026-11-01)

| Date | Commit | Type | Change | Metric | Before → after | Δ |
|---|---|---|---|---|---|---|
| 2026-11-01 | `67d76c1` | fix | run the full pipeline from the repository root, not from src/chess_lab | — | — | — |
| 2026-11-01 | `cbc067a` | feat | keepalive script that restarts the laptop wifi when the worker connection drops | — | — | — |

Internal, not itemised: 10 docs, 6 refactor, 1 test (`319b8b6`…`a8bff93`).

## 3.0.0 (2026-10-31)

| Date | Commit | Type | Change | Metric | Before → after | Δ |
|---|---|---|---|---|---|---|
| 2026-10-31 | `bcf2169` | feat | worker matches run detached with stop, resume, status and fetch, per-engine options | — | — | — |
| 2026-10-31 | `1a12cdb` | feat | remote worker scripts to deploy the UCI binary and run fastchess matches over SSH | — | — | — |
| 2026-10-31 | `fa2f45c` | feat | opening suite generator for engine matches, balanced positions filtered with Stockfish | — | — | — |
| 2026-10-31 | `f972d78` | feat | bench command with fixed positions and depth, deterministic node counts | — | — | — |
| 2026-10-31 | `0f920bc` | feat | uci binary with go nodes/movetime/clock, stop and setoption, node limit per iteration | — | — | — |
| 2026-10-31 | `111867e` | feat | the clock decides the search depth (no default max_depth, min_depth 1) | — | — | — |

Internal, not itemised: 13 refactor, 3 docs, 1 chore (`59b9db8`…`6162db8`).

## 2.1.0 (2026-10-30)

| Date | Commit | Type | Change | Metric | Before → after | Δ |
|---|---|---|---|---|---|---|
| 2026-10-30 | `7c6d1be` | fix | Keras models in the window search with one thread | — | — | — |
| 2026-10-30 | `fc4723d` | feat | player selection, AI settings dialog and main window; main.py opens the GUI | — | — | — |
| 2026-10-30 | `3e9efa7` | feat | Qt board, game page with clocks and move list, cburnett pieces | — | — | — |
| 2026-10-30 | `c49c872` | feat | game model and AI settings of the GUI, without Qt | — | — | — |
| 2026-10-30 | `635704a` | fix | get_info of the model bot works without an opening book | — | — | — |
| 2026-10-30 | `e3c9655` | fix | list the incremental technique in Techniques::NAMES; drop stale port comments | — | — | — |

Internal, not itemised: 4 refactor, 1 docs (`f9b97e3`…`6055277`).

## 2.0.0 (2026-10-26)

| Date | Commit | Type | Change | Metric | Before → after | Δ |
|---|---|---|---|---|---|---|
| 2026-10-26 | `ae86998` | fix | unknown --model exits with code 2 and lists the models; help no longer mentions the tournament mode | — | — | — |
| 2026-10-26 | `c928842` | feat | tournament records carry the core version and a file is not resumed with another one | — | — | — |
| 2026-10-26 | `e73be3e` | feat | main.py plays you against the default bot; the duplicate tournament mode is removed | — | — | — |
| 2026-10-25 | `f5e368a` | feat | null move and LMR; the bots use all techniques but lmr by default (+205 Elo in 100 games), measurements keep the reference search | Elo of the bots with techniques vs the reference search, 100 games<br>Elo SPRT [-5,0] without null move, 710 games 10+0.1 | +205 (76.5 %)<br>+97.4 ± 24.5 (H1: keep) | —<br>— |
| 2026-10-25 | `8c274f0` | feat | tt_move, history, pvs and aspiration techniques (each behind its switch; values equal to the reference) | Elo SPRT [-5,0] without tt_move, 1,098 games 10+0.1 | +56.5 ± 18.3 (H1: keep) | — |
| 2026-10-25 | `c6a7e0e` | feat | incremental accumulator for net4_pst (technique 'incremental'), same values up to rounding | Elo SPRT [-5,0] without incremental, 1,690 games 10+0.1 | +35.5 ± 14.7 (H1: keep) | — |
| 2026-10-25 | `6efb56a` | feat | engine-version comparison script and technique suffix in player specs; ablation script covers the selective techniques | — | — | — |
| 2026-10-25 | `5fef49d` | feat | switches for six selective-search techniques (all off = reference) and the real ply in the search | — | not measurable (first Rust port, no comparable command) | — |
| 2026-10-25 | `b65f80f` | feat | helper threads search ahead of the main thread; thread scaling measured; docs of the Rust move | — | not measurable (first Rust port, no comparable command) | — |
| 2026-10-25 | `9a998c3` | feat | Lazy SMP threads with a lock-free shared table, book and tablebase delegate to the core, CI builds Rust, speed measurement script | — | — | — |
| 2026-10-25 | `e6c45cb` | feat | the Rust core lists legal moves in the order python-chess generates them; goldens made by the unmodified Python engine; recorded games replay move for move | — | not measurable (first Rust port, no comparable command) | — |
| 2026-10-25 | `67c556c` | feat | the search bots run the Rust core (evaluation native for h1, linear models and the network); ablation switches in the core | — | — | — |
| 2026-10-25 | `940ba2c` | feat | Rust opening book and endgame tables, Python bindings (chess_engine._core), maturin build; the core matches every golden file | — | not measurable (first Rust port, no comparable command) | — |
| 2026-10-25 | `006f394` | feat | Rust alpha-beta search with iterative deepening, transposition table, killers and quiescence; matches the Python golden searches node for node | times faster than the Python search | 8-35x (same moves and nodes) | — |
| 2026-10-25 | `1558f4f` | feat | Rust core with the rules, the h1 heuristic, linear and network evaluations (not wired yet) | — | not measurable (first Rust port, no comparable command) | — |

Internal, not itemised: 7 docs, 4 test, 3 refactor, 1 chore (`148b86d`…`1f00b91`).

## 1.1.0 (2026-10-22)

| Date | Commit | Type | Change | Metric | Before → after | Δ |
|---|---|---|---|---|---|---|
| 2026-10-22 | `8b49014` | feat | the opening book and the endgame tables are part of the search bots unless switched off; measurement players switch them off | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-22 | `2ecb50f` | feat | opening book as an option of every search bot (--opening-book), off by default | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-22 | `ef364f6` | fix | the model bot defaults to linear_pst_terms, which ships with the repository | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `f939f30` | feat | opening book of 121,868 positions, checked by sampling against Stockfish | book entries within 50 cp of the best line (2,438 sampled) | 98.7 % | — |
| 2026-10-21 | `6dc604a` | feat | the book check can use a local UCI engine instead of the Lichess service | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `8270725` | fix | the book check waits out rate limits and resumes from saved answers | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `aeb9f0c` | feat | script that checks a sample of the opening book against the Lichess cloud evaluation | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `d3e848c` | feat | the default estimator of the next iteration is geometric_growth; tournaments record and check it (--estimator) | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `9ae5227` | feat | the opening book is built from the Lichess evaluations (23,239 positions, 16 plies) | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `c0c9c42` | feat | the leakage check also compares accuracy with and without a neighbour in train | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `3ad5962` | feat | script that checks the split for duplicates, mirror images and neighbours | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `4f22ca5` | feat | the estimator comparison also reports the moves that overrun the budget | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `19c859e` | feat | script that builds the one opening book from the Lichess evaluations | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `820aa32` | feat | script that measures the conversion of won endgames with and without tablebases | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `2fc0f86` | feat | optional Syzygy endgame tablebases for the search bots (--endgame-db); the bots accept every search option | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `1dd5d78` | feat | script that replays the stopping rules of the deepening on recorded iteration times | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `d53b64a` | feat | script that records the duration of every iteration in replayed game positions | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `349ec4b` | feat | the five models retrained at effort 30 with 10 % of the mates; the variant models and their cards are removed | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `a27aa4b` | feat | per-iteration times and stop reason in the search statistics; monotonic clock; selectable estimator of the next iteration | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `b6c63fd` | feat | script that summarizes the search depth per move from tournament logs | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `e14cf12` | feat | script that measures what each search technique saves | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `0cdf404` | fix | iterative deepening always completes depth 3; the time estimate could stop at depth 2 | moves with the iteration stopped at depth 2 (10 min + 5 s), % | 12 → 0 | -100.0 % |
| 2026-10-21 | `b1581f7` | feat | versioned quiescence comparison script | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `85beb50` | feat | the card keeps one tournament entry per time control | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `a33fa8e` | feat | retraining clears the measurements of the previous model; evaluate accepts h1 | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `9bbf2d0` | fix | truthful CLI help, Ctrl-C leaves the human player, drop unused bot helpers | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-21 | `b0cb157` | feat | skip the calibration when the number of openings is given | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-19 | `347b0ba` | feat | --max-depth for the tournament; depth-matched result explains the speed advantage | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-18 | `1dd8cb8` | feat | learning-rate decay on plateau, --lr-decay; duration estimate counts the cache | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-18 | `8ec1034` | feat | cache encoded training batches on disk | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-18 | `a403fd6` | feat | 10 % of the mate positions is the default training | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-18 | `fe7723e` | feat | quiescence search on by default for search bots (4 plies) | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-17 | `40571a4` | feat | optional quiescence search at the horizon | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-17 | `ec28d04` | feat | --seed for the TRAIN sample of the training CLI | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-17 | `5656f3d` | feat | effort is one million positions per point | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-17 | `fa412d6` | feat | effort is a percentage of the training rows | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-17 | `6a765d6` | feat | tournament report shows how the games ended | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-17 | `989dfd4` | feat | per-move logs for every tournament game | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-17 | `db4f659` | other | tournament defaults include nn4 (registered) | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-17 | `9c2095c` | feat | sparse-input neural evaluation model nn4 | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-17 | `d8086ef` | other | night-1 model list, skip unregistered models, explicit profile and tournament players | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-17 | `6a9de62` | fix | perspective-aware slotted entry bound, bound checks on random positions | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-17 | `3639b41` | feat | slotted and king-bucket features, king-bucket linear model | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-17 | `5b8f9b9` | fix | tournament defaults drop the unregistered nn8 player | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-17 | `519dbfe` | feat | h3t, the tuned classical reference, and the h3t-anchored tournament defaults | — | not measurable (Python engine, no command reproduces it) | — |

Internal, not itemised: 24 docs, 9 refactor, 5 chore, 4 results, 3 test, 2 data, 2 style (`1ab1001`…`eea7a02`).

## 1.0.0 (2026-10-16)

| Date | Commit | Type | Change | Metric | Before → after | Δ |
|---|---|---|---|---|---|---|
| 2026-10-16 | `d1e0849` | fix | bake-off tag prefix, refuse to overwrite existing tagged models | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `27a9f12` | feat | representation bake-off script | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `2c3f555` | fix | mate-keep guard, mate-aware row count, tagged evaluate, rebuild reset on played games | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `5e62519` | feat | profile and tournament results are stored in the model cards | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `2cfe7db` | feat | mate-undersampled model variants and tagged model specs | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `36d36d7` | feat | model cards and history table for the trade-off comparison | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `9ac6254` | feat | piece-square tables plus classical terms model | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `e3c7a77` | feat | effort-based training for linear models: validation-aware estimate, trial tests | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `8c61b72` | feat | material linear model; remove linear_baseline | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `0a3a06b` | fix | requeue games of a dead pool, repair partial tails, reject bad script args | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `ee879a2` | feat | effort-based training for linear models | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `02c29a3` | feat | judge with rank correlation and best-move agreement by slice | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `3d4f8d4` | fix | harden tournament against crashes, truncated files and mixed time controls | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `25c6df1` | fix | validate validation loss, parallel featurization, clearer load and base-class errors | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `44d4eaf` | feat | linear evaluation core and tapered piece-square-table model | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `a942b79` | feat | parallel round-robin tournament | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `9023bbd` | fix | report skipped players, single-thread parallel workers in profiler | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `db28627` | fix | bounded-memory load_sample, n=0 batches, join-consistency test | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `7420f29` | feat | depth-accuracy profiler at fixed time per move | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `55a866d` | feat | dataset sidecar with split and quiet flag, sampler | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `ca04d9e` | other | Convert a local Lichess dump in parallel, drop illegal positions | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `ecd13ed` | feat | record the deepest completed iteration | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `fff8eff` | feat | data representation and training targets | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `574eb32` | other | Raw Lichess evaluations in Parquet: fetch script and 1M-position trial | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `ea86b20` | other | Fast transposition key and make/unmake in the advanced search | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `ba3d5cb` | other | Fast terminal detection with the same draw rules; search benchmark | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `2fd8de7` | other | Move the code into a src/chess_engine package and the docs into docs/ | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `6ad36e6` | other | Engine tests; record forfeits in PGN; compute SAN before pushing the move | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `21f6f60` | other | Package evaluation and knowledge; lint | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `f30d04c` | other | Fix model signs in the shipped CNN weights, NumPy MLP inference, strength match, more tests, lint-clean | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `ab20c20` | other | Refactor bots onto shared bases; restore tactical bot; CLI --model/--depth; normalize line endings | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `d8b3fd1` | other | Add models and training packages: extractors, registry, adapters, pipeline CLIs | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `f470fdf` | other | Move features/heuristics to an evaluation package; add OpeningBook | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `1ee37d6` | other | Refactor advanced search: shared helpers, bounded TT, faster-win bonus and truncation handling | — | not measurable (Python engine, no command reproduces it) | — |
| 2026-10-16 | `c35ef31` | other | Import from coursework (cleaned) | — | not measurable (Python engine, no command reproduces it) | — |

Internal, not itemised: 2 refactor (`a72312e`…`baad7d7`).
