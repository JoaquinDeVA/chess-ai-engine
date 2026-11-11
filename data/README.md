# data/

What lives here and what can be deleted. Only this file and `knowledge/opening_book/` are versioned.

## Permanent (keep)

* `knowledge/opening_book/opening_book.json` – the one opening book of the project, used by `IterativeModelBookBot`
  (`{fen: uci_move}`), versioned. Built from the Lichess evaluations with
  `python -m chess_lab.datasets.build_opening_book`.
* `knowledge/endgame_tablebase/` – the Syzygy endgame tablebases for 3, 4 and 5 pieces (290 files, 939 MB:
  the `.rtbw` win/draw/loss tables and the `.rtbz` distance-to-zeroing tables). **Not versioned**
  (too large) but not regenerable either: copy them back from `tablebase.lichess.ovh` or from
  any other Syzygy 3-4-5 set. Bots use them with `python main.py --endgame-db`; see
  `docs/techniques/endgame-tablebases.md`.

## Training data (can be rebuilt, can be deleted)

* `LichessPositions/raw/` – raw sharded Parquet export of the Lichess evaluations (126 shards, 124.9 M
  positions). **Not versioned**. Created with:

  ```bash
  python -m chess_lab.datasets.build_raw_shards --input data/LichessPositions/lichess_db_eval.jsonl.zst
  ```
* `LichessPositions/sidecar/` – one aligned sidecar per shard (split, quiet flag, side to move). **Not
  versioned**. Created with:

  ```bash
  python -m chess_lab.datasets.build_sidecars --raw data/LichessPositions/raw --out data/LichessPositions/sidecar --jobs 4
  ```
* `cache/` – encoded training batches (`docs/TRAINING.md`, "The cache of encoded
  batches"). Safe to delete; the next training rebuilds what it needs.

## Results of the runs (can be deleted)

Everything under `results/`:

* `results/matches/` – one folder per fastchess match, with `match.log` and `match.pgn`.
* `results/tournament/` – the games of the tournaments and their per-move logs.
* `results/reports/` – the generated reports (tournaments, trade-off, search measurements) and `diag/`, loose diagnostic logs.
* `results/pipeline/` – state and logs of `src/chess_lab/pipeline.sh`.

Schema, sign convention and commands: `docs/DATA.md`.
