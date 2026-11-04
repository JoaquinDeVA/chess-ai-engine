# Training

> **What it is:** What `--effort` means, what a training run does, and how to reproduce the whole flow.
>
> **Read it when:** You want to train a model or rerun the pipeline.
>
> **Next:** [Models](MODELS.md)

The data this page trains on is described in [Training data](DATA.md).

## The effort scale

One number sets how much work a training run does. `--effort N` means N million training positions, with a floor of 1 million, and `1 + N // 10` epochs (an epoch is one pass over the positions). From effort 40 the run also searches hyper-parameters: `1 + (N - 40) // 15` random trials of the L2 penalty and the learning rate, each fitted on a tenth of the positions. Effort goes from 0 to 100, and the position count never exceeds the training rows available (`effort_config` in `src/chess_lab/training/effort.py`).

| Effort | Positions | Epochs | Trials |
|---|---|---|---|
| 15 | 15 M | 2 | 0 |
| 30 | 30 M | 4 | 0 |
| 55 | 55 M | 6 | 2 |

The rows follow from the formulas above. For example, effort 30 trains four passes over 30 million positions with fixed default hyper-parameters:

```bash
python -m chess_lab.training.train --model linear_pst_terms --effort 30
```

The shipped `linear_pst_terms` is trained on quiet positions only, at effort 10; the filter is not the default of `Representation` (`"all"`, `src/chess_lab/models/representation.py`), so the command passes it ([Models](MODELS.md#piece-square-and-terms-logistic-model-linear_pst_terms)):

```bash
python -m chess_lab.training.train --model linear_pst_terms --effort 10 --representation '{"filter":"quiet"}'
```

Before training, the command prints the configuration and an estimated duration, based on a short timed run on 50,000 positions.

## What training does

The linear models score a position as a weighted sum of features (material, piece-square tables and so on), pass the sum through the logistic function, and compare the result with the soft target. The loss is binary cross-entropy against targets that may be any number from 0 to 1, not just 0 or 1 (`losses` in `src/chess_lab/models/linear_eval.py`). Adam, a standard gradient-descent variant, updates the weights, with an L2 penalty that keeps them small.

After each epoch the model is scored on a validation sample of up to 50,000 positions (`MAX_VALIDATION_ROWS` in `src/chess_lab/training/train.py`). The weights of the epoch with the lowest validation loss are the ones saved. Two ways of stopping are available (`run_epochs` in `src/chess_lab/models/linear_eval.py`):

- Default: stop after more than `patience` (2) epochs without improvement.
- With `--lr-decay F` (a factor between 0 and 1, for example 0.5): each epoch without improvement multiplies the learning rate by F and training continues, until the rate falls below 1e-5 or the epochs run out.

With `--patience N` as well, the run also stops after N epochs in a row without improvement (the rate is still cut on each of them); `--epochs` sets the cap. Without `--lr-decay`, `--patience N` replaces the default of 2 (the run stops after more than N epochs without improvement).

```bash
python -m chess_lab.training.train --model linear_pst_terms --effort 30 --lr-decay 0.5
python -m chess_lab.training.train --model linear_pst_terms --effort 30 --lr-decay 0.5 --patience 3 --epochs 40
```

Training writes the model's card (`src/chess_lab/models/cards/`); retraining first empties the measurements of
the previous model (`clear_measurements` in `src/chess_lab/models/cards.py`), so a card never mixes the numbers of
two models.

### The cache of encoded batches

Turning a FEN into features is the slow part, and it is identical in every epoch. So the first epoch stores each encoded batch on disk (`data/cache/linear`) and later epochs read them back. The cache directory is a short hash of everything that shapes the batches: model class, representation name (which includes the mate share), number of positions, seed, quiet filter, mate share, batch size and a cache version (`cache_path` in `src/chess_lab/training/effort.py`). Change any of them and a new directory is used.

The gain was measured while developing it, on 500,000 rows with the linear model: 133.6 s for an epoch that encodes, against 2.6 s for an epoch served from the cache, about 50 times faster. The 2.6 s figure is the constant `CACHED_SECONDS_PER_ROW` in `src/chess_lab/training/effort.py`, used for the duration estimate. Pass `--no-cache` to encode every epoch.

## Reproducing everything

All commands run from the repository root, after `pip install -e ".[dev]"`. The data directory takes about 24 GB.

Step 1 downloads the database (resumable, four connections because Lichess answers 429 to more) and converts it to shards. Each shard is written to a temporary file and renamed, so it exists complete or not at all, and repeating the command skips finished shards.

```bash
aria2c -x 4 -s 4 -k 64M --continue=true --auto-file-renaming=false --file-allocation=none \
    --max-tries=0 --retry-wait=30 -d data/lichess -o lichess_db_eval.jsonl.zst \
    https://database.lichess.org/lichess_db_eval.jsonl.zst
python src/chess_lab/scripts/data/build_raw_shards.py --input data/lichess/lichess_db_eval.jsonl.zst \
    --output-dir data/lichess/raw --log-file data/lichess/progress.log
```

Useful options of `build_raw_shards.py`: `--rows` (only the first N lines), `--shard-rows` (default 1,000,000), `--workers`, and `--url` instead of `--input` to stream over HTTP, which is slower and more fragile. The command above converts the whole file; to reproduce the 126 shards used here, stop after 126,000,000 lines with `--rows 126000000`.

Step 2 builds the sidecars; shards that already have one are skipped:

```bash
python src/chess_lab/scripts/data/build_sidecars.py --raw data/lichess/raw --out data/lichess/sidecar --jobs 4
```

Step 3 trains a model. `--seed` changes the training sample, `--raw` and `--sidecar` point to other directories, and `--output` saves elsewhere:

```bash
python -m chess_lab.training.train --model linear_pst_terms --effort 15
```

The split check and the opening book are run by hand:

```bash
python src/chess_lab/scripts/data/check_split_leakage.py --models linear_pst_terms
python src/chess_lab/scripts/data/build_opening_book.py --out data/opening_book.json
```

The whole flow, from sidecars to training, evaluation and the results table, is `src/chess_lab/scripts/run_full_pipeline.sh`. Each step runs once and leaves a `.done` marker; it trains every registered model unless `MODELS` names some. Preview it without running anything, then launch it in the background:

```bash
DRY_RUN=1 bash src/chess_lab/scripts/run_full_pipeline.sh
nohup setsid bash src/chess_lab/scripts/run_full_pipeline.sh > data/pipeline.log 2>&1 &
```

It trains five models at effort 30 by default (`EFFORT` and `MODELS` change that). Results and the commands that measure them are in [Results](RESULTS.md); the models themselves are in [Models](MODELS.md).
