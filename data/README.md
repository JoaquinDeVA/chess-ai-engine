# data/

* `opening_book.json` – opening book used by `IterativeModelBookBot` (`{fen: uci_move}`), versioned.
* `lichess/raw/` – raw sharded Parquet export of the Lichess evaluations (126 shards, 124.9 M
  positions). **Not versioned**. Created with:

  ```bash
  python scripts/data/fetch_lichess_raw.py --input data/lichess/lichess_db_eval.jsonl.zst
  ```
* `lichess/sidecar/` – one aligned sidecar per shard (split, quiet flag, side to move). **Not
  versioned**. Created with:

  ```bash
  python scripts/data/prepare_dataset.py --raw data/lichess/raw --out data/lichess/sidecar --jobs 4
  ```

Schema, sign convention and commands in `docs/RESULTS.md` (section "Datos"). The CSV of the old pipeline
(`evaluations.csv`) has been retired.
