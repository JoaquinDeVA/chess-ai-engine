# Training tasks

Checklist to (re)train and benchmark the models and refresh `MODELS_HISTORY.md`.

**Rule:** one model per *approach*. Retraining or tweaking an approach overwrites its entry
(`models/cards/<name>.json`); a new name is only added for a genuinely different approach.

## Run everything (unattended)

- [ ] `data/evaluations.csv` exists. Either of:
  - `ln -s /path/to/evaluations.csv data/evaluations.csv`
  - `python scripts/download_lichess_evals.py --rows 1000000` (public Lichess database, CC0)
- [ ] `bash scripts/run_all.sh` (about 1–2 h on CPU; add `GAMES=0` to skip the matches)
  - trains `linear_baseline`, `ridge_features`, `mlp_features` (minutes)
  - benchmarks all five models: error and agreement with Stockfish, time per evaluation
  - plays the strength matches (10 games against H2 and H3)
  - regenerates the table in `MODELS_HISTORY.md` and the plots in `docs/img/`
- [ ] Look at `MODELS_HISTORY.md` and `docs/img/*.png`, then `git add src/chess_engine/models docs && git commit`.

## Optional

- [ ] `TRAIN_CNN=1 bash scripts/run_all.sh` retrains `cnn_resnet` (needs a GPU: ~2 min per epoch).
- [ ] Label your own positions: `python scripts/label_with_stockfish.py --random 50000 --depth 12`.

## Adding a new approach

- [ ] Follow `docs/ADDING_A_MODEL.md` (a model is one small file plus a config).
- [ ] `python -m chess_engine.training.train --model <name> --data data/evaluations.csv`
- [ ] `python -m chess_engine.training.evaluate --model <name> --data data/evaluations.csv --strength`
- [ ] `python scripts/plot_models.py` and commit.
