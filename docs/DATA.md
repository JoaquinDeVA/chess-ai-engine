# Training data

> **What it is:** Where the training data comes from and how a Stockfish evaluation becomes a number a model can learn.
>
> **Read it when:** You want to rebuild the data or know why the mate positions are undersampled.
>
> **Next:** [Training](TRAINING.md)

The engine's learned evaluators answer one question: "how good is this position for the player to move?". Nobody labelled positions by hand. Instead, the project borrows the answers of Stockfish, the strongest open-source chess engine, from a public database, and trains small models to imitate them. This page follows that path: raw database, Parquet shards, sidecars and targets; [Training](TRAINING.md) takes it from there.

## The Lichess evaluation database

Lichess publishes `lichess_db_eval.jsonl.zst` (<https://database.lichess.org/lichess_db_eval.jsonl.zst>, CC0 licence, 22.4 GB compressed). It holds 416,442,401 positions evaluated with Stockfish, one JSON line each: the position as a FEN string (a one-line text description of a chess position) and a list of evaluations of that position, each with its search depth, node count and principal variations. These figures come from `docs/RESULTS.md`, section "Datos".

The project does not use all of it. The file is not shuffled: the deepest and most balanced positions come first, so any prefix is a biased sample of the whole. The project reads the first 126,000,000 lines, drops the illegal positions and keeps 124,923,027 records, about 30 % of the database. The training sampler then does the shuffling itself (see "Sidecars"), so the bias of the file order does not reach the models. The line count and the illegal count come from summing the shard metadata:

```bash
python -c "
import glob, pyarrow.parquet as pq
meta = [pq.ParquetFile(p).metadata.metadata for p in glob.glob('data/lichess/raw/part-*.parquet')]
print(sum(int(m[b'lines']) for m in meta), sum(int(m[b'illegal']) for m in meta))
"
```

It prints `126000000 1076973`; the difference is the 124,923,027 kept rows (`docs/RESULTS.md`, "Datos"). Only legal standard-chess positions pass `chess.Board(fen).is_valid()`.

## Raw shards

`src/chess_lab/scripts/data/build_raw_shards.py` converts the `.zst` file into 126 Parquet shards of 1,000,000 lines each (`data/lichess/raw/part-NNNNN.parquet`, ignored by git). Parquet is a columnar file format that reads quickly and compresses well. Each row keeps the deepest evaluation of the position, with ties broken by node count; nothing is rewritten, so the shard is a lossless copy of what Stockfish said (`src/chess_lab/training/lichess_raw.py`, `RAW_SCHEMA`).

| Column | Type | Meaning |
|---|---|---|
| `fen` | string | The position, four FEN fields (no move counters) |
| `cp` | int16, nullable | Centipawns of the best line (a pawn is 100 centipawns) |
| `mate` | int16, nullable | Moves to forced mate; exactly one of `cp` and `mate` is set |
| `depth` | int16 | Search depth of the chosen evaluation |
| `knodes` | int32 | Thousands of nodes searched by that evaluation |
| `best_move` | string | First move of the best line, in UCI notation |
| `n_evals`, `n_pvs` | uint8 | Evaluations of the line, and variations of the chosen one |

The sign convention matters. In the shards, `cp` and `mate` are always from White's point of view: positive is good for White, and `mate > 0` means White delivers mate, whoever is to move. The code measures this instead of assuming it (`detect_perspective` in `src/chess_lab/training/lichess.py`) and writes it in each shard's metadata under the `perspective` key. Training converts the target to the side to move: the score of a position is `1 - s` when Black is to move (`target` in `src/chess_lab/models/representation.py`). One model therefore serves both colours.

## Sidecars

Next to each shard sits a small file, `data/lichess/sidecar/part-NNNNN.sidecar.parquet`, with the same rows in the same order and the same row groups. It stores three things per position (`src/chess_lab/training/sidecar.py`):

- `split`: train, validation or test. It is the CRC32 hash of the seed (42) and the FEN, modulo 10,000, so 80 % goes to train, 10 % to validation and 10 % to test (`assign_split` in `src/chess_lab/training/dataset.py`). The split depends only on the position, never on row order or on how many rows you load, so a model can never be tested on a position it trained on.
- `quiet`: true when the side to move is not in check and the best move is neither a capture nor a promotion (`is_quiet`). Quiet positions are the ones where a static evaluation is least distorted by an exchange in progress.
- `white_to_move`: whose turn it is, so the sampler does not need to parse the FEN again.

The sampler draws positions with a seeded random generator, `numpy.random.default_rng(seed)` with seed 42 (`SAMPLE_SEED` in `src/chess_lab/models/defaults.py`), shuffling the shard order and the order inside each row group. The same seed gives the same positions. Pairing a shard with its sidecar means only the columns and row groups needed are read.

| Measure | Value |
|---|---|
| Shards / rows | 126 / 124,923,027 |
| Train / validation / test | 99,935,706 / 12,496,155 / 12,491,166 |
| Quiet positions | 0.693 of the rows |
| White to move | 0.510 of the rows |

All four rows come from `docs/RESULTS.md`, section "Datos". To rebuild the sidecars, run the command in "Reproducing everything".

## Is the test set clean?

The split is by position, so a position (and any repeat of it) is in one split only. Two different
positions that are almost the same can still sit on both sides, and the data has no game identifiers
to split by game. `python src/chess_lab/scripts/data/check_split_leakage.py --models <models>` measures it on the
whole data (it indexes all 124.9 million positions) and on 20,000 test positions:

| check | result |
|---|---|
| positions that appear more than once in the shards | 0 |
| test positions whose colour-swapped image is in the data (in train) | 1.1 % (1.0 %) |
| test positions with a position one move away in the data (in train) | 72.0 % (60.2 %) |

The last row is a lower bound (only the positions reached by one more move are searched). It is
inherent to splitting positions that come from real games: most positions have a neighbour. To see
whether the neighbours flatter the models, the fixed test sample of 200,000 positions is split into
the 59.6 % with a neighbour in train and the rest, and each model is judged on both:

| model | Spearman with a neighbour | Spearman without | MAE with | MAE without |
|---|---|---|---|---|
| `linear_material` | 0.493 | 0.446 | 0.1541 | 0.1607 |
| `linear_pst` | 0.643 | 0.599 | 0.1318 | 0.1406 |
| `linear_king_bucket` | 0.629 | 0.588 | 0.1377 | 0.1452 |
| `linear_pst_terms` | 0.665 | 0.623 | 0.1283 | 0.1372 |
| `net4_pst` | 0.714 | 0.679 | 0.1168 | 0.1237 |

Every model does better with a neighbour, including the material model, which has six weights and
cannot memorize anything: the difference (0.035 to 0.047 in Spearman) comes from the kind of position
that has neighbours, which are more ordinary, and not from copying. The network, the model with the
most capacity, gains the least (0.035). Nothing here suggests that the accuracy of the models is
inflated by the split. The training and validation losses agree as well (the gap is at most 0.003,
`training` section of each model card).

## From centipawns to a target

Stockfish speaks in centipawns, but a model that learns to predict centipawns directly is dominated by lopsided positions: a queen up is +900 and drowns out the quiet fights that decide most games. A +900 position is not "nine times better" than a +100 one; both are mostly about who wins. So the target is a win probability between 0 and 1, the logistic curve Lichess itself uses:

    target = 1 / (1 + exp(-k * cp)),   k = 0.00368208

With this curve, +100 centipawns maps to about 0.59 and +900 to about 0.96: the scale is bounded and saturates. Forced mates become exactly 1 (White mates) or 0 (Black mates). Both come from `white_score` in `src/chess_lab/models/representation.py`, with `LICHESS_K = 0.00368208`; the two values quoted above come from evaluating that formula.

### Why only 10 % of the mates

Mate positions all share the same target (exactly 1 or 0) whatever the material on the board, and the database holds many of them. By default training keeps 10 % of them (`DEFAULT_MATE_KEEP = 0.10` in `src/chess_lab/models/defaults.py`). The kept set is chosen by a hash of the FEN (`keep_mate_rows`), so it is the same on every run and a smaller share is always a subset of a larger one. The share applies to the train and validation sets.

The evidence is one match, played once. `linear_pst_terms` trained with 10 % of the mates played 100 games against the same model trained with all of them: effort 15, 50 openings with colours swapped, 60 s + 1 s per move. The 10 % model scored 71.0 % (64 wins, 14 draws, 22 losses; 95 % interval 62.9 to 79.1), which is +156 Elo (`docs/RESULTS.md`, section "Mates al 10 %"). It is a single pair of players at a single effort level, not a general law.

The old 100 % model was retired from the repository; it lives in git commit `03169db` (`src/chess_engine/models/artifacts/linear_pst_terms.npz` and its card). To train the 100 % variant again:

```bash
python -m chess_lab.training.train --model linear_pst_terms --effort 15 --mate-keep 1.0 --tag mates100
```

A mate share other than the default requires `--tag`, so that it cannot overwrite the untagged model (`train_by_effort` in `src/chess_lab/training/train.py`).
