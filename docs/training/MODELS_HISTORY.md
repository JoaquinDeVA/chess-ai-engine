# Models

One row per model. The table is generated from `models/cards/*.json` by
`python -m chess_engine.comparison.report`; `bash scripts/run_all.sh` regenerates the cards and the
table (see [TRAINING_TASKS.md](TRAINING_TASKS.md)). A dash means the model has no result yet for
that column.

Columns:

- **Kind**: `plays` models are trained at an effort level and take part in the round-robin;
  `accuracy record` models are the older ones. Their current weights are measured, but they are
  neither retrained nor played.
- **Params** and **Representation** (target scale, point of view, position filter) and **Effort**
  (0-100, see [ADDING_A_MODEL.md](../ADDING_A_MODEL.md)).
- **Spearman**, **MAE**: rank correlation and mean absolute error of the expected score against
  Stockfish, on the fixed test sample (200,000 positions, seed 42). MAE is in expected-score units,
  from 0 to 1.
- **Best move, all / non-quiet**: share of positions where one search step picks the Stockfish best
  move (the first 20,000 sample positions, or fewer when `--agreement-n` is lower).
- **µs per eval**: median time of one evaluation of one position, as the search calls it.
- **Depth 1 s / 3 s / 10 s**: mean completed search depth with a fixed time per move, and **search
  agreement** at 10 s: how often that search finds the Stockfish best move.
- **Score**, **Elo ± interval**: round-robin score and Elo with its 95% interval; the full results
  are in `TOURNAMENT.md`.

<!-- BEGIN GENERATED TABLES -->
| Model | Kind | Params | Representation | Effort | Spearman | MAE | Best move, all / non-quiet | µs per eval | Depth 1 s | Depth 3 s | Depth 10 s | Search agreement, 10 s | Score | Elo ± interval |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cnn_light` | accuracy record | 109,569 | — | — | — | — | — | — | — | — | — | — | — | — |
| `cnn_resnet` | accuracy record | 572,353 | — | — | 0.371 | 0.165 | 20.3% / 40.4% | 24,407.1 | — | — | — | — | — | — |
| `mlp_features` | accuracy record | 1,505 | — | — | 0.614 | 0.130 | 17.9% / 43.6% | 124.2 | — | — | — | — | — | — |
| `ridge_features` | accuracy record | 30 | — | — | — | — | — | — | — | — | — | — | — | — |
<!-- END GENERATED TABLES -->
