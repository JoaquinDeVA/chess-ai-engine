# Results

> **What it is:** What was measured, what was only estimated, which model plays best, and the command or file behind every number.
>
> **Read it when:** You want to compare models or techniques, check a claim of the other pages, or repeat a measurement.
>

All numbers are from this project, on a six-core CPU (see Training data for the positions the models
are trained on). Commands are run from the repository root. Two engines produced them: the Python engine, until
2026-10-25, and the Rust engine since (The search in Rust); every section says which
one. Scripts under `scripts/bench/` and `play_tournament` were removed in `6162db8` and are in the git history.

## The short answer

- **The engine that plays is the Rust UCI binary** with seven search techniques on by default. Four won an SPRT
  against the set without them: incremental accumulator, table move, null move and static exchange evaluation
  (+35.5 to +97.4 Elo, "Round robin" below). `lmr`, `rfp` and `iir` were switched on later by their node count at
  a fixed depth (`uci ttd 11`, see the [changelog](../CHANGELOG.md)); their SPRT is queued. An earlier `lmr` lost
  8.6 ± 6.1 Elo at 20,000 nodes per move (`lmr_nodes` below), so that conflict is still open. Five others (`pvs`,
  `aspiration`, `history`, `history_malus`, `qs_tt`) reached no verdict and were retired in `efb595e`.
- **The model that plays is the Piece-square network, width 64 (`net_pst_w64`)**, the default of the binary since
  `4d8dfc4`. There is no SPRT of it against `net_pst_w32`, the model it replaced; that one beat the
  Piece-square-and-terms logistic model (`linear_pst_terms@e30`) by +165.1 ± 32.9 Elo (SPRT [0, 5], 504 games at
  10 s + 0.1 s), and the logistic model was retired.
- **More accurate is not stronger**: the hybrid networks, with the classical terms as inputs, have a higher
  Spearman (0.702 to 0.722 against 0.693) but search about half the nodes per second and lose
  (`net_hybrid_w32` against `net_pst_w32`, −105.5 ± 25.5 Elo, H0); see Models.

## Measured and estimated

| Claim | How it is known | Engine | Strength of the evidence |
|---|---|---|---|
| The measuring instrument has no bias | Three A/A matches of 400 to 500 games, fastchess ([Measurement](MEASUREMENT.md#the-aa-check)) | Rust | Measured; intervals ±24 to ±27 Elo |
| Each of the four default techniques makes the engine stronger | SPRT [−5, 0] removing it from the default set of then, 710 to 1,690 games at 10 s + 0.1 s ("Round robin" below) | Rust | Measured; H1 accepted for each. The set of four has no SPRT against the set of seven |
| `pvs`, `aspiration`, `history`, `history_malus` and `qs_tt` help | Same round robin, 2,134 to 2,200 games each | Rust | Not shown: no verdict (LLR 0.19 to 2.21); retired |
| The network plays better than the logistic model | SPRT [0, 5] of 504 games at 10 s + 0.1 s, +165.1 ± 32.9 Elo | Rust | Measured; H1 accepted |
| Adding the classical terms to the network makes it weaker | SPRT [0, 5] of 672 games at 10 s + 0.1 s, −105.5 ± 25.5 Elo | Rust | Measured; H0 accepted, but it mixes quality and speed (the hybrids have no accumulator). At fixed nodes: pending |
| Late move reductions made the engine weaker (removed) | SPRT of 10,244 games at 20,000 nodes per move: 48.76 %, −8.6 ± 6.1 Elo | Rust | Measured; H0 accepted |
| Several threads search faster | Time to depth 8 on 60 positions (Lazy SMP) | Rust | Measured speed; effect in games not measured |
| Training on quiet positions makes the model stronger | SPRT of 362 games at 60 s + 1 s, +284 ± 43 Elo (logistic model) | Rust | Measured; H1 accepted |
| Quiescence, endgame tables and the opening book help | Position suites ("Decisions that are settled") | Python | Measured on positions, not in games |

## Round robin (Rust engine, fastchess)

From 2026-11-06 to 2026-11-09 the compute node played every job of `src/chess_lab/scripts/worker/queue.json` at
`d2d03ba` in turns (`src/chess_lab/scripts/worker/roundrobin.py`): 10 s + 0.1 s, `Hash` 16, one thread, the 10,000
openings of `data/openings/openings.epd`, LLR bounds ±2.94. Technique jobs used the default set of then
(`incremental,tt_move,history,pvs,aspiration,null_move,see`) and `linear_pst_terms`; `abl_` removes one switch from
it (A is the full set, so a positive Elo is what the switch gains), `add_` adds one. Model jobs used the default
techniques. It was stopped on 2026-11-09 with jobs still running; their figures are the last report. Files:
`data/results/roundrobin/<job>/llr.csv`.

| job | A against B | SPRT | games | Elo (95 %) | LLR | verdict |
|---|---|---|---|---|---|---|
| `abl_null_move` | default against default without `null_move` | [−5, 0] | 710 | +97.4 ± 24.5 | +2.95 | H1: keep |
| `abl_see` | default against default without `see` | [−5, 0] | 992 | +64.5 ± 19.5 | +2.96 | H1: keep |
| `abl_tt_move` | default against default without `tt_move` | [−5, 0] | 1,098 | +56.5 ± 18.3 | +2.96 | H1: keep |
| `abl_incremental` | default against default without `incremental` | [−5, 0] | 1,690 | +35.5 ± 14.7 | +2.96 | H1: keep |
| `abl_pvs` | default against default without `pvs` | [−5, 0] | 2,200 | +18.2 ± 12.1 | +2.21 | no verdict; retired |
| `abl_history` | default against default without `history` | [−5, 0] | 2,200 | +17.5 ± 12.4 | +2.09 | no verdict; retired |
| `abl_aspiration` | default against default without `aspiration` | [−5, 0] | 2,200 | +10.7 ± 12.5 | +1.37 | no verdict; retired |
| `add_history_malus` | default + `history_malus` against default | [−5, 0] | 2,200 | +9.2 ± 12.0 | +1.24 | no verdict; retired |
| `add_qs_tt` | default + `qs_tt` against default | [−5, 0] | 2,134 | −0.3 ± 12.5 | +0.19 | no verdict; retired |
| `model_net_vs_linear` | `net_pst_w32` against `linear_pst_terms@e30` | [0, 5] | 504 | +165.1 ± 32.9 | +2.95 | H1: stronger |
| `model_width_64_vs_32` | `net_hybrid_w64` against `net_hybrid_w32` | [0, 5] | 1,612 | +44.0 ± 15.9 | +2.95 | H1: stronger |
| `model_terms_in_net_nodes` | `net_hybrid_w64` against `net_pst_w32` at 250,000 nodes per move | [0, 5] | 784 | +89.2 ± 22.5 | +2.95 | H1: stronger per node |

The SPRTs `best_net_hybrid_w64_vs_linear`, `best_net_hybrid_k4_w64_vs_linear`, `model_terms_in_net` and
`model_buckets_k8_vs_k4` were removed from this table: they ran at a time control before the incremental accumulator
of the hybrid and king-bucket networks (2026-11-09), against networks that had it, so they mixed quality with speed. They
are replaced by `acc_*` and `pst_*` jobs with the same binary for both sides (`queue.json` in `src/chess_lab/scripts/worker/`).
The files stay in `data/results/roundrobin/`.

## Engine matches (Rust engine, fastchess)

Earlier matches of the UCI binary against itself, one change at a time, as explained in
[How a change is measured](MEASUREMENT.md). Logs in `data/results/matches/<name>/match.log`.

| match | A against B | conditions | games | score | Elo (95 %) | LLR | verdict |
|---|---|---|---|---|---|---|---|
| `aa_s11` | default against default | 10 s + 0.1 s, SPRT [−5, 0] | 400 counted | 50.5 % | +3.5 ± 26.8 | +0.11 | no bias |
| `aa_s12` | default against default | 10 s + 0.1 s, SPRT [−5, 0] | 500 | 51.3 % | +9.0 ± 25.0 | +0.28 | no bias |
| `aa_s13` | default against default | 10 s + 0.1 s, SPRT [−5, 0] | 500 | 50.7 % | +4.9 ± 23.9 | +0.18 | no bias |
| `lmr_nodes` | default + `lmr` against default | 20,000 nodes per move, SPRT [−5, 0] | 10,244 | 48.76 % | −8.6 ± 6.1 | −2.94 | H0: regresses |
| `qs_evasions` | default + `qs_evasions` against default | 20,000 nodes per move, SPRT [−5, 0] | 3,428 | 46.94 % | −21.3 ± 10.6 | −2.96 | H0: regresses |
| `see_nodes` | default + `see` against default | 20,000 nodes per move, SPRT [0, 5] | 1,188 | 58.04 % | +56.4 ± 17.5 | +2.97 | H1: stronger (fixed nodes) |
| `history_malus_nodes` | default + `history_malus` against default | 20,000 nodes per move, SPRT [0, 5] | 2,404 | 50.56 % | +3.9 ± 9.3 | +0.33 | no verdict (stopped by hand at 2,556 games; Elo as of the last report, 2,404 games) |
| `qs_tt` | default + `qs_tt` against default | 20,000 nodes per move, SPRT [−5, 0] | 10,000 | 50.10 % | +0.7 ± 3.9 | +1.55 | no verdict (round cap) |
| `quiet_vs_actual` | `linear_pst_terms` trained on quiet positions against the same trained on all positions | 60 s + 1 s, SPRT [0, 5] | 362 | 83.70 % | +284 ± 43 | +2.96 | H1: stronger |
| `q_k65_vs_quiet` | quiet model with `k` 0.0065 against the quiet model | 60 s + 1 s, SPRT [0, 5] | 280 | 48.93 % | −7 ± 35 | −0.13 | no verdict (round cap) |
| `q_m05_vs_quiet` | quiet model with 0.5 % of the mates against the quiet model | 60 s + 1 s, SPRT [0, 5] | 280 | 43.75 % | −44 ± 36 | −0.59 | no verdict (round cap) |
| `q30_vs_quiet` | quiet model with 30 M positions against the quiet model | 60 s + 1 s, SPRT [0, 5] | 240 | 53.33 % | +23 ± 41 | +0.22 | no verdict (stopped by hand) |
| `reloj_nuevo` | new clock against the old clock, same model and techniques | 60 s + 1 s, SPRT [0, 5] | 720 | 40.83 % | −64 ± 24 | −2.06 | no verdict (stopped by hand); the new clock is worse |
| `see_vs_base` | default + `see` against default | 10 s + 0.1 s, 300 games (screening) | 300 | 56.83 % | +47.8 ± 34.3 | - | screening only |
| `see_sprt` | default + `see` against default | 60 s + 1 s, SPRT [0, 5] | 160 | 55.94 % | +41.5 ± 46.1 | +0.30 | no verdict (stopped by hand) |
| `see_sprt10` | default + `see` against default | 60 s + 1 s, SPRT [0, 10] | 282 | 57.27 % | +50.9 ± 38.6 | +1.13 | no verdict (stopped by hand at 303 games; Elo as of the last report, 282 games) |
| `passed_vs_base` | `linear_pst_terms` plus passed-pawn features against `linear_pst_terms` | 10 s + 0.1 s, 300 games (screening) | 300 | 54.17 % | +29.0 ± 35.2 | - | screening only |
| `king_vs_base` | `linear_pst_terms` plus passed-pawn and king-attack features against `linear_pst_terms` | 10 s + 0.1 s, 300 games (screening) | 300 | 50.00 % | 0.0 ± 35.3 | - | screening only |
| `e30_vs_base` | `linear_pst_terms` with 30 M positions (`linear_pst_terms@e30`) against `linear_pst_terms` | 10 s + 0.1 s, 300 games (screening) | 300 | 55.67 % | +39.6 ± 33.8 | - | screening only |

## How the models are measured

| Judge | What it measures | Command |
|---|---|---|
| Static | Spearman correlation and error of the model against Stockfish on 200,000 test positions (seed 42), scored by the Rust core as the engine plays it (clipped to 0.01–0.99); µs per eval is the median of one core evaluation from a FEN over 1,000 positions | `python -m chess_lab.measure.evaluate --model net_pst_w32@e30` |
| Match | Games of the UCI binary against itself with fastchess, Elo with a 95 % interval | [Measurement](MEASUREMENT.md) |

## Search techniques in games

The Rust engine reaches 2 to 3 plies more than the Python one in the same time (The search in Rust). The six techniques on by default before
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
| Is the opening book good? | 98.7 % of 2,438 sampled moves are within 50 centipawns of Stockfish's best line, none illegal. | `python src/chess_lab/scripts/data/check_opening_book.py data/opening_book.json --fraction 0.02 --engine /usr/games/stockfish --depth 16` |

## Limits and leakage

- **Positions, not games.** The data has no game identifiers, so the split is by a hash of the position.
  `python src/chess_lab/scripts/data/check_split_leakage.py` found no duplicate position among the 124,923,027
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
| Model | Params | Representation | Effort | Spearman | MAE | Best move, all / non-quiet | µs per eval |
|---|---:|---|---:|---:|---:|---:|---:|
| `net_hybrid_k4_w64` | 198,145 | winprob-k0.00368-stm-quiet | 30 | 0.722 | 0.110 | 22.0% / 56.8% | 3.3 |
| `net_hybrid_k8_w64` | 394,753 | winprob-k0.00368-stm-quiet | 30 | 0.719 | 0.111 | 21.9% / 56.5% | 3.3 |
| `net_hybrid_w32` | 25,345 | winprob-k0.00368-stm-quiet | 30 | 0.702 | 0.115 | 21.4% / 56.2% | 2.8 |
| `net_hybrid_w64` | 50,689 | winprob-k0.00368-stm-quiet | 30 | 0.718 | 0.111 | 21.7% / 56.5% | 2.9 |
| `net_pst_k4_w64` | 196,737 | winprob-k0.00368-stm-quiet | 30 | 0.709 | 0.114 | 21.7% / 56.6% | 2.4 |
| `net_pst_w32` | 24,641 | winprob-k0.00368-stm-quiet | 30 | 0.693 | 0.118 | 21.3% / 56.2% | 2.4 |
| `net_pst_w64` | 49,281 | winprob-k0.00368-stm-quiet | 30 | 0.712 | 0.113 | 21.8% / 57.2% | 2.4 |
| `net_pst_w64x16` | 50,273 | winprob-k0.00368-stm-quiet | 30 | 0.716 | 0.110 | 21.5% / 55.8% | 2.4 |
| `h1` | 6 | — | — | 0.561 | 0.149 | 19.6% / 49.8% | 2.8 |
| `linear_pst_terms` (retired) | 791 | winprob-k0.00368-stm-quiet | 30 | 0.646 | 0.128 | 20.8% / 56.7% | 2.4 |
<!-- END GENERATED TABLES -->

## Reproduce

The pipeline rebuilds the sidecars, trains the models, evaluates them and regenerates the model table; it plays no
games (games are played with fastchess, [Measurement](MEASUREMENT.md)).

```bash
DRY_RUN=1 bash src/chess_lab/scripts/run_full_pipeline.sh   # prints every command
nohup setsid bash src/chess_lab/scripts/run_full_pipeline.sh > data/results/pipeline/pipeline.log 2>&1 &
```
