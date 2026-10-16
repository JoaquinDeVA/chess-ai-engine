# Training and evaluation redesign (phases 4 and 5)

Status: draft for approval · 2026-10-16

## Why

The engine's learned evaluations come from a university course (`docs/training/MODELS_HISTORY.md`).
They were trained on ~1M course positions with ad-hoc targets (centipawns clipped at ±1267,
White's point of view), and the useful ones are either slow (the CNNs: 50-80 nodes/s, 7-50 s per
move at depth 3) or bound by expensive features (`mlp_features` spends ~22% of the search computing
its 29 inputs, mostly counting legal moves). The goal of the bot is to **search deeper in the same
time without playing worse**, so the evaluation must be cheap per node and trained on much more and
better-prepared data.

The course feedback (notebooks in the 2nd-year AI course folder) still applies and motivates several
decisions: the dataset is dominated by near-zero evaluations; the cp scale used in training must
match the one used when playing; mirrored-vertical augmentation produces illegal positions; a deep
model is in tension with search depth; and candidate models should be compared systematically, with
hyper-parameters chosen by search rather than by hand.

## Goals

1. Decide, by experiment, how raw Stockfish evaluations are turned into training data
   (target scale, point of view, position filter), and keep only that processed version.
2. Add a fast, non-neural evaluation family and train it on millions of positions.
3. Make training a single command per model with an **effort level 0-100** that chooses data size,
   epochs and hyper-parameter search, and prints a time estimate first (an overnight run is fine).
4. A comparison (phase 5) that ranks models by what matters for play: accuracy, cost per
   evaluation, depth reached in fixed time, and results in games at equal time.
5. Keep only versions (datasets, models) that have a measured reason to exist.

## Non-goals

- Changing the search algorithm (quiescence, LMR, null move, better ordering). Phase 2 only made
  nodes cheaper; deeper-search techniques are a later project.
- New neural networks, retraining the existing ones, or NNUE-style evaluation.
- Opening book or endgame tablebases.
- GPU training.

## Anti-goals (a failure even if everything "works")

- A model that is more accurate against Stockfish but makes the bot reach less depth and lose more
  games at equal time.
- Train and play disagreeing on the scale or the point of view of the evaluation (the course bug:
  trained with 1267, played with 20000).
- Several processed copies of the dataset, or models kept "just in case", without a measured reason.
- A comparison mixing changes: each phase is measured on its own (search changes were measured
  with the evaluation fixed; model changes are measured with the search fixed).

## Constraints

- Machine: 6 CPU cores, no GPU, 22 GB RAM (often ~16 GB already in use), plenty of disk.
- Data budget: ~10 GB on disk for the Lichess data (raw + processed).
- Python 3.12, python-chess, numpy / scikit-learn / Keras already in the project. New dependencies
  only with a reason (pyarrow is being added for Parquet).
- Existing interfaces stay: `EvaluationModel` and the model registry (`docs/ADDING_A_MODEL.md`),
  bots discovered by the registry, model cards in `src/chess_engine/models/cards/`, generated table
  in `docs/training/MODELS_HISTORY.md`, the FEN-hash train/validation/test split.
- The search is the one after phase 2 (fast terminal detection, fast TT key, make/unmake). Its
  benchmark is `scripts/benchmark_search.py`.
- Implementation is delegated to subagents task by task; decisions that need judgment (reading a
  bake-off, choosing a variant) stay with the coordinator and the user.

## Phase 3 input: raw data (in progress)

A trial run stores the first 1M positions of the Lichess evaluation database
(`lichess_db_eval.jsonl.zst`, CC0, 416M positions, 22.4 GB compressed) as raw Parquet under
`data/lichess/raw/`, one row per position: `fen`, `cp` or `mate` (as in the database, sign
convention recorded), `depth`, `knodes`, `best_move`, `n_evals`, `n_pvs`. The measurements and
statistics go to `docs/training/DATA.md`. **The amount to keep for the full run is decided from
those numbers** (size per position, throughput, whether the start of the file is representative),
within the 10 GB budget. The raw version is kept because every target/filter choice is derived from
it and re-downloading costs 22 GB.

## Phase 4A: choose the data representation (bake-off)

One cheap model, many variants, a fixed judge. The model is the piece-square-table model (2 below):
it trains in minutes and is the main candidate, so the choice is made where it matters most.

Training set: ~5M positions from the train split; evaluation on a fixed sample of the test split.

Variants (full factorial is not required; start from the default and change one axis at a time):

| Axis | Options | Default |
|---|---|---|
| Target scale | (a) cp clipped to ±1267, linear; (b) win probability with Lichess's curve `1 / (1 + exp(-0.00368208·cp))`; (c) same sigmoid with its scale fitted to the data | (b) |
| Point of view | White; side to move (board mirrored when Black moves) | side to move |
| Filter | all positions; quiet only (side to move not in check and best move not a capture or promotion) | all |
| Mates | win/loss probability ~1/0 (with the margin the search uses) for mates within N; longer mates dropped or treated as large cp | ~1/0, N decided from the data |

The judge must not depend on the target scale, so the variants are compared with:

1. Spearman rank correlation between the model and Stockfish on the test sample.
2. Best-move agreement: share of test positions where a 1-ply search (and, if cheap enough, 2-ply)
   with the model picks Stockfish's `best_move`.
3. Tie-breaker: short matches at fixed depth between the top variants.

*Coordinator's call:* the test sample contains **all** positions, whatever the training filter, and
every metric is also reported on two slices: quiet and non-quiet. The search has no quiescence
search, so its leaves are often tactical; a quiet-trained model that is worse on the non-quiet
slice is a real risk the judge must be able to see. The quiet filter is a variant, not the default.

*Coordinator's call (process):* the pipeline produces all the evidence in one go: metrics 1-2 for
every variant, then matches between the two best variants by best-move agreement. Then it stops:
the coordinator reads the results and the user confirms the choice. Decisive match results beat
the static metrics.

Outcome: one processed representation, written once (e.g. `data/lichess/processed/`), documented
in `DATA.md`. The other variants' data is deleted; the results table stays in the documentation.

## Phase 4B: models and effort-based training

### Model family (new, non-neural, linear in their parameters)

1. **Material** - tuned value per piece type. Reference point for "how much does the rest add".
2. **Piece-square tables, tapered** - value of each piece type on each square, in a middlegame and
   an endgame table, blended by game phase (non-pawn material). From the side to move's point of
   view the tables of both colours are one table mirrored, plus a side-to-move (tempo) term. Main
   candidate: it can be updated per move in O(1).
3. **Tables + classical terms** - model 2 plus pawn structure (passed, doubled, isolated pawns),
   bishop pair, rooks on open / half-open files, king pawn shield, mobility from attack masks (not
   from legal-move generation). Still linear, so trained the same way. Answers "is more knowledge
   per node worth fewer nodes per second".

All three output a logit; the expected score is its sigmoid.

*Coordinator's call (scale contract):* the search consumes an **expected score in
`[margin, 1 - margin]` for a given player** (the `(state, player) -> float` convention of the
heuristics `h1`-`h4` and of `ModelEvaluator`; terminal values are 0 / 0.5 / 1). `EvaluationModel`
gains one method that every model implements, the expected score of the side to move (or of
White, converted inside the model), and `ModelEvaluator` converts it to the root player.
Each model converts from its own training target inside the model, with the constants saved in
the artifact/card next to the weights, so training and playing cannot disagree. Existing models
keep their centipawn path internally; `predict_cp` stays for the metrics against Stockfish
(converted with the chosen curve). Phase 5 compares all models in the expected-score scale. Training is logistic regression on the
chosen target ("Texel tuning"), mini-batch, streaming from Parquet so data size is not bound by RAM.

Evaluation speed is part of the deliverable: each model gets a direct `board -> score` path that
does not go through numpy per position. A first version computes the score from scratch (O(pieces));
**incremental update during make/unmake is a follow-up decided by phase 5's measurements** (only if
evaluation cost still dominates the search).

### Existing models (user's decision: accuracy record only)

`ridge_features`, `mlp_features`, `cnn_light`, `cnn_resnet` are too slow to finish most games, but
they are valuable as a record of how closely a model can fit Stockfish. They are **not retrained,
do not play matches and are not subject to the deletion rule**: phase 5 measures them, with their
current weights, on the same test set and metrics as the new models (accuracy and cost per
evaluation), so their fit can be compared. `linear_baseline` is superseded by model 1 (material) and
removed once model 1 exists.

### Effort level 0-100

`python -m chess_engine.training.train --model <name> --effort <0-100>` (plus the existing options).
Each model class maps the effort to its own configuration:

- positions: log scale from ~100k (effort 0) to everything available (100);
- maximum epochs and early-stopping patience;
- hyper-parameter search: none at low effort, a growing number of random-search trials at high
  effort (validation split decides);
- the command prints the chosen configuration and an estimated duration before starting
  (estimated from a short timed run), and records the effort and the real duration in the card.

Effort 100 for everything is allowed to take a night (~10 h).

## Phase 5: comparison

For every model, written to its card and to the generated table in `MODELS_HISTORY.md`:

| Group | Metric |
|---|---|
| Accuracy | error in the common scale (expected score), Spearman correlation, best-move agreement |
| Cost | microseconds per evaluation (single position, as the search calls it) |
| In search | nodes per second and depth reached in a fixed time per move, on the benchmark positions |
| Play | W-D-L at equal time against `IterativeHeuristicBot` with `h2` and `h3`, and round-robin between models; paired openings with colours swapped |

Models in the comparison: the new 1 (material, reference), 2 and 3 play; the existing models only
get the accuracy and cost metrics (see 4B).

Match budget (user's decision): **one night, ~10 h**. Fast time control (order of 10 s + 0.1 s
per move), several games in parallel; every model first plays H2/H3, and only the best ones play a
round-robin between themselves.

Decision rule (user's decision): **strength at equal time decides.** The strongest model stays;
another one survives only if it wins clearly on something that matters (e.g. much faster with
similar strength). Everything else is removed (artifact, card, config, code if unused).
`TRAINING_TASKS.md` and `scripts/run_all.sh` are updated to the new workflow.

## Order of work

1. Phase 3 trial (running) -> decide the full download -> full download.
2. 4A: processed-data pipeline for the variants + model 2 + judge -> bake-off -> choose.
3. 4B: models 1 and 3, effort levels.
4. Phase 5: comparison tooling, runs, cleanup.

The full-download size (prefix of the file vs. a sample streamed across the whole file) is decided
by the coordinator and the user from the trial numbers, including whether the start of the file is
representative. "Everything available" in the effort scale means whatever that download keeps.

Steps that do not depend on each other (e.g. writing models 1 and 3 while the full download runs)
run in parallel.

## Left to the builder

- Module and file layout inside `src/chess_engine/{training,models}` and the Parquet layout of the
  processed data.
- The exact phase formula, term definitions of model 3, regularization, optimizer and batch size.
- The exact effort-to-configuration curves, as long as they are monotonic and documented.
- Sample sizes for the judge and the match protocol (enough games to separate clear differences;
  report the uncertainty).
