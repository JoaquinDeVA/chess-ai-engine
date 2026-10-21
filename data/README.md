# data/

What lives here and what can be deleted. Only this file and `opening_book.json` are versioned.

## Permanent (keep)

* `opening_book.json` – the one opening book of the project, used by `IterativeModelBookBot`
  (`{fen: uci_move}`), versioned. Built from the Lichess evaluations with
  `python scripts/data/build_opening_book.py`.
* `knowledge/endgame/` – the Syzygy endgame tablebases for 3, 4 and 5 pieces (290 files, 939 MB:
  the `.rtbw` win/draw/loss tables and the `.rtbz` distance-to-zeroing tables). **Not versioned**
  (too large) but not regenerable either: copy them back from `tablebase.lichess.ovh` or from
  any other Syzygy 3-4-5 set. Bots use them with `python main.py --endgame-db`; see
  `docs/ALGORITHMS.md`.

## Training data (can be rebuilt, can be deleted)

* `lichess/raw/` – raw sharded Parquet export of the Lichess evaluations (126 shards, 124.9 M
  positions). **Not versioned**. Created with:

  ```bash
  python scripts/data/build_raw_shards.py --input data/lichess/lichess_db_eval.jsonl.zst
  ```
* `lichess/sidecar/` – one aligned sidecar per shard (split, quiet flag, side to move). **Not
  versioned**. Created with:

  ```bash
  python scripts/data/build_sidecars.py --raw data/lichess/raw --out data/lichess/sidecar --jobs 4
  ```
* `cache/linear/` – encoded training batches (`docs/DATA_AND_TRAINING.md`, "The cache of encoded
  batches"). Safe to delete; the next training rebuilds what it needs.

## Results of the runs (can be deleted)

* `pipeline/` – state and logs of `scripts/run_full_pipeline.sh`.
* `tournament/` – the games of the tournaments and their per-move logs.
* `reports/` – the generated reports (tournaments, trade-off, search measurements).

Schema, sign convention and commands: `docs/DATA_AND_TRAINING.md`.
