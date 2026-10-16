# Models

One row per *approach*: retraining or tuning a model overwrites its row, and a new row is only added
for a genuinely different approach. The table is generated from `models/cards/*.json` by
`python -m training.report`; regenerate everything with `bash scripts/run_all.sh`
(see [TRAINING_TASKS.md](TRAINING_TASKS.md)).

Columns: architecture, size, training data, **time per evaluation** (features included, one position at
a time, CPU), **error against Stockfish** in centipawns, **agreement** (model and Stockfish agree on who
is better, ±50 cp counting as equal), share of predictions within 50 / 100 cp, and playing strength as
`W-D-L` of an ML bot using the model against H2 and H3 at equal depth.

> The evaluation protocol and the models themselves are being redesigned; see
> [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md) and [ROADMAP.md](ROADMAP.md). Treat numbers as relative.

<!-- BEGIN GENERATED TABLES -->
| Model | Status | Architecture | Params | Size | Training data | Time / eval | MAE (cp) | RMSE (cp) | Agreement | ±50 cp | ±100 cp | Playing strength |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| `cnn_light` | current | CNN (5 conv layers 32-32-64-64-64, GAP) + dense branch -> Dense32-16-1 | 109,569 | 496 KB | course dataset (Stockfish evaluations, mirror-augmented) | — | — | — | — | — | — | — |
| `cnn_resnet` | current | ResNet-style CNN (Conv32-64, two dilated residual blocks, Conv256, GAP) + dense branch on 9 extras -> Dense256-128-64-1 | 572,353 | 2.3 MB | course dataset (Stockfish evaluations, mirror-augmented) | — | — | — | — | — | — | — |
| `linear_baseline` | current | LinearRegression on 6 material features (turn + piece-count differences) | 7 | 1 KB | 745,515 positions, evaluations.csv, 1 epochs | — | — | — | — | — | — | — |
| `mlp_features` | current | MLP 29 -> 32 -> 16 -> 1 (ReLU), NumPy inference | 1,505 | 45 KB | 745,515 positions, evaluations.csv, 40 epochs | — | — | — | — | — | — | — |
| `ridge_features` | current | StandardScaler -> Ridge(alpha=10) on 29 features | 30 | 2 KB | 745,515 positions, evaluations.csv, 1 epochs | — | — | — | — | — | — | — |
<!-- END GENERATED TABLES -->
