# Raw Lichess evaluation data

## Source

Public Lichess evaluation database, `lichess_db_eval.jsonl.zst`
(<https://database.lichess.org/lichess_db_eval.jsonl.zst>), CC0. 22.4 GB compressed,
416,442,401 positions evaluated with Stockfish. One JSON line per position:

```json
{"fen": "8/4r3/2R2pk1/6pp/3P4/6P1/5K1P/8 b - -",
 "evals": [{"pvs": [{"cp": 0, "line": "e7a7 f2e3 ..."}, {"mate": 3, "line": "..."}],
            "knodes": 491568, "depth": 58}, ...]}
```

FENs have no move counters. `pvs[0]` is the principal variation; the other PVs of the same
evaluation may mix `cp` and `mate`. `evals` holds several evaluations of the same position
(different depths/nodes).

## Raw dataset (`data/lichess/raw/part-NNNNN.parquet`)

One row per database line, built from the evaluation with the greatest `depth` (ties: greater
`knodes`). Values are stored exactly as in the database (no sign change). Git-ignored.

| column | dtype | meaning |
|---|---|---|
| `fen` | string | position as in the DB (4 fields, no counters) |
| `cp` | int16, null | centipawns of `pvs[0]` (range seen: -20000..20000, i.e. clamped) |
| `mate` | int16, null | mate distance of `pvs[0]`; exactly one of `cp`/`mate` is set |
| `depth` | int16 | depth of the chosen evaluation (max seen 245) |
| `knodes` | int32 | thousands of nodes of the chosen evaluation |
| `best_move` | string, null | first move (UCI) of `pvs[0].line` |
| `n_evals` | uint8 | evaluations the line had |
| `n_pvs` | uint8 | PVs of the chosen evaluation |

Parquet, zstd, row groups of 100,000 rows. File metadata (JSON values): `perspective`, `lines`,
`kept`, `dropped`.

**Sign convention.** `perspective` is detected from the data
(`chess_engine.training.lichess.detect_perspective`, material correlation on Black-to-move
positions). Result on the trial: **`white`**, i.e. `cp`/`mate` are from White's point of view
(positive = good for White, `mate > 0` = White mates) whoever is to move. `data/evaluations.csv`
(old pipeline, unchanged) uses the same convention.

Code: `src/chess_engine/training/lichess_raw.py` (parser, shard writer),
`scripts/data/fetch_lichess_raw.py`.

## Running

The streaming of the whole file over one HTTP connection is slow (about 1 MB/s) and fragile, so
the file is downloaded first with `aria2c` (resumable) and then converted from the local copy:

```bash
aria2c -x 4 -s 4 -k 64M --continue=true --auto-file-renaming=false --file-allocation=none \
    --max-tries=0 --retry-wait=30 -d data/lichess -o lichess_db_eval.jsonl.zst \
    https://database.lichess.org/lichess_db_eval.jsonl.zst
python scripts/data/fetch_lichess_raw.py --input data/lichess/lichess_db_eval.jsonl.zst \
    --output-dir data/lichess/raw --log-file data/lichess/progress.log
```

The downloaded `lichess_db_eval.jsonl.zst` (22,423,777,007 bytes) is a **temporary file**: delete
it after the conversion has finished and the shards are checked. Lichess answers HTTP 429 when
too many connections are opened at once (16 were refused), hence 4.

Options: `--input` (local `.zst`) or `--url` (stream over HTTP, the default), `--rows` (database
lines to read, the first ones; without it the whole file is converted), `--output-dir`,
`--shard-rows` (lines per shard, default 1,000,000), `--workers` (parsing processes, default
cores - 2), `--log-file` and `--progress-seconds` (a progress line with lines, compressed
GB read/total, percentage, MB/s, ETA and counters is printed and appended to the log every 60 s).
The compressed file is never decompressed to disk. Only legal standard-chess positions are kept
(`chess.Board(fen).is_valid()`; Chess960 setups, more than 32 pieces, ...); they are counted
separately from unusable lines (no valid evaluation) in the logs and in the shard metadata
(`dropped`, `illegal`). Building the board is the expensive part (about 22,000 FENs/s per core;
the validity checks themselves 210,000/s), so parsing and the check run in a pool of processes
while the main process decompresses; memory stays bounded (about 200 MB in the main process).
Shards are written to a `.tmp` file and renamed, so a shard exists complete or not at all.
Restarting with the same options skips the finished shards (their lines are decompressed but not
parsed). Network stalls on `--url` raise after 60 s; with `--input` there is no network. If the
stream ends before `--rows` lines the exit code is 1.

## Trial run: first 1,000,000 positions

| measurement | value |
|---|---|
| lines read / kept / dropped | 1,000,000 / 1,000,000 / 0 |
| wall time | 88.6 s (CPU 19.4 s: network-bound on this connection) |
| compressed downloaded | 87.4 MB (0.99 MB/s) |
| parse+write rate | 11,300 lines/s wall; about 51,000 lines/s CPU-only |
| Parquet size | 22.2 MB = 22.2 bytes/position |
| peak memory | 223 MB |
| duplicate FENs | 0 |
| perspective | white |

Note: 87.4 MB per million is 4x the 22.4 GB / 416M average (53.8 MB per million), so the first
positions are heavier (more evaluations, deeper) than the rest of the file.

### Statistics (1M rows)

| | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| depth | 14 | 17 | 22 | 28 | 38 | 190 | 245 | 245 |
| knodes | 7 | 578 | 4,664 | 15,541 | 57,932 | 635,258 | 4,163,971 | 248,024,081 |
| cp (non-mate) | -857 | -438 | -20 | 14 | 89 | 609 | 1,532 | 20,000 |
| mate distance (abs) | 1 | 1 | 3 | 6 | 12 | 24 | 35 | 70 |
| pieces on board (incl. kings) | 3 | 4 | 13 | 24 | 29 | 32 | 32 | 63* |

\* more than 32 pieces: the DB contains some odd/illegal-looking positions; not investigated.

| depth | <=19 | 20-29 | 30-39 | 40-49 | 50-59 | 60-69 | >=70 |
|---|---|---|---|---|---|---|---|
| share | 11.1% | 42.9% | 22.0% | 7.0% | 3.4% | 1.7% | 7.0% |

| share | value |
|---|---|
| mate (rest cp) | 17.7% (81% of mates are positive, i.e. White mates) |
| non-mate with abs(cp) <= 50 / 100 / 300 / 1000 | 47.3% / 63.8% / 79.6% / 97.6% (2.4% beyond 1000) |
| White to move | 52.1% |
| side to move in check | 7.9% |
| `best_move` is a capture / promotion | 21.1% / 0.4% |
| `best_move` missing | 0% |
| pieces <=6 / 7-10 / 11-16 / 17-24 / 25-32 | 14.7% / 5.9% / 9.7% / 21.7% / 47.9% |

`n_evals`: 1: 58.5%, 2: 19.8%, 3: 14.8%, 4: 5.9%, 5+: 1.0%.
`n_pvs`: 1: 56.4%, 2: 9.2%, 3: 16.9%, 4: 2.9%, 5: 14.6%, others <0.2% (max 50).

### Ordering / representativeness

Per block of 100k rows there is a clear drift along the file: mean depth falls 51.3 -> 40.8,
median abs(cp) rises 49 -> 62, mate share 18.8% -> 17.2% (min 15.0% at block 7), median
knodes 22.6k -> 12.8k (first vs last 100k), mean `n_evals` 1.88 -> 1.63, share with
abs(cp) <= 100 67.7% -> 61.3%. Mean number of pieces and side to move are stable (21.2 /
52.7% vs 21.2 / 51.3%). So the file is not randomly ordered: deeper, better-evaluated and
more balanced positions come first. The first 1M rows are not a uniform sample of the DB.

## Extrapolation to the full database (416.4M positions)

| | estimate |
|---|---|
| Parquet size at 22.2 B/position | 9.2 GB (fits, barely, in 10 GB; later rows have fewer evals so likely smaller) |
| positions in 10 GB | about 450M (all of them) |
| compressed bytes to read | 22.4 GB (official size) |
| time at 1 MB/s (this connection) | about 6.2 h (network-bound) |
| time if network were not the limit | about 2.3 h of CPU (51k lines/s) |
| memory | about 220 MB |

## From raw shards to training

1. `scripts/data/fetch_lichess_raw.py` writes the shards (above).
2. `scripts/data/prepare_dataset.py` writes one sidecar per shard (`part-NNNNN.sidecar.parquet`,
   same rows and row groups as the shard). Each position gets its split (train, validation or
   test, from a seeded hash of its FEN, `chess_engine.training.dataset.assign_split`), its quiet
   flag (not in check and the best move is neither a capture nor a promotion) and its side to move.
   Shards that already have a sidecar are skipped.
3. `chess_engine.training.train` streams the training rows from the sidecars and the raw shards.
   `chess_engine.comparison.evaluate` judges a model on a seeded sample of 200,000 test rows
   (seed 42).

The models trained on these data and their results are in [MODELS_HISTORY.md](MODELS_HISTORY.md).
