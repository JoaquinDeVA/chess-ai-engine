# data/

* `opening_book.json` – opening book used by `AdvancedMLBot` (`{fen: uci_move}`), versioned.
* `evaluations.csv` – Stockfish evaluations used to train and benchmark the evaluation models
  (columns `fen, cp, is_mate, mate_distance`, White's point of view). **Not versioned**; create it
  with one of:

  ```bash
  python scripts/download_lichess_evals.py --rows 1000000        # public Lichess database (CC0)
  python scripts/label_with_stockfish.py --random 50000          # your own positions
  ```
