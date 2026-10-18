# Models

One row per model. The table is generated from `models/cards/*.json` by
`python -m chess_engine.comparison.report`; `bash scripts/run_all.sh` regenerates the cards and the
table (see `TRAINING_TASKS.md`, plan fuera del repo). A dash means the model has no result yet for
that column.

Columns:

- **Kind**: `plays` models are trained at an effort level and take part in the round-robin;
  `accuracy record` models are the older ones (`cnn_resnet`, `mlp_features`). Their current weights are measured, but they are
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
| `linear_king_bucket` | plays | 10,369 | winprob-k0.00368-stm-all | 15 | 0.609 | 0.143 | 17.9% / 47.1% | 35.2 | — | — | — | — | 59.0% | +134 ± 67 |
| `linear_material` | plays | 6 | winprob-k0.00368-stm-all | 15 | 0.475 | 0.156 | 19.1% / 54.9% | 21.0 | — | — | — | — | 69.0% | +198 ± 72 |
| `linear_pst` | plays | 769 | winprob-k0.00368-stm-all | 15 | 0.623 | 0.137 | 18.8% / 50.6% | 57.2 | — | — | — | — | 48.5% | +70 ± 66 |
| `linear_pst_terms` | plays | 791 | winprob-k0.00368-stm-all | 15 | 0.655 | 0.129 | 19.3% / 51.7% | 116.6 | — | — | — | — | 37.5% | +3 ± 66 |
| `linear_pst_terms@mates10` | plays | 791 | winprob-k0.00368-stm-all-mates10 | 15 | 0.646 | 0.133 | 20.0% / 54.4% | 114.7 | — | — | — | — | — | — |
| `linear_pst_terms@mates10s7` | plays | 791 | winprob-k0.00368-stm-all-mates10 | 15 | 0.650 | 0.132 | 20.2% / 54.8% | 118.5 | — | — | — | — | — | — |
| `linear_pst_terms@quiet` | plays | 791 | winprob-k0.00368-stm-quiet | 15 | 0.651 | 0.128 | 19.8% / 53.5% | 117.1 | — | — | — | — | — | — |
| `linear_pst_terms@s7` | plays | 791 | winprob-k0.00368-stm-all | 15 | 0.645 | 0.131 | 18.4% / 49.1% | 115.2 | — | — | — | — | — | — |
| `net4_pst` | plays | 206,177 | winprob-k0.00368-stm-all | 15 | 0.627 | 0.133 | 18.9% / 48.1% | 68.4 | — | — | — | — | 49.0% | +73 ± 64 |
| `cnn_resnet` | accuracy record | 572,353 | — | — | 0.371 | 0.165 | 20.3% / 40.4% | 24,407.1 | — | — | — | — | — | — |
| `mlp_features` | accuracy record | 1,505 | — | — | 0.614 | 0.130 | 17.9% / 43.6% | 124.2 | — | — | — | — | — | — |
<!-- END GENERATED TABLES -->
