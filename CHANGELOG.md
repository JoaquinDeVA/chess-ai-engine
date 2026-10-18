# Changelog

## Unreleased

- Training only reads the raw shards (`--effort`); the CSV path (`--data`, `load_evaluations`,
  `scripts/data/download_lichess_evals.py`, `scripts/data/label_with_stockfish.py`) is removed.
- One epoch loop for the linear and network models (`run_epochs` in `models/linear_eval.py`).
- Shared defaults in `training/spec.py` (sample seed, mate share, name suffix).
- The generated tables go to `docs/RESULTS.md` (was a `MODELS_HISTORY.md` that no longer existed); the
  tournament, trade-off and bake-off reports are written to `data/reports/`.
- Measured `linear_material` at effort 90: no gain over effort 15 (see `docs/RESULTS.md`).

- `h1` replaces the heuristics `h2`, `h3` and `h4`: the features of the old `h3` with weights fitted to
  Stockfish (`h1_weights.json`, `scripts/data/fit_h1.py`). `h1` is the Elo anchor.
- Removed the models `ridge_features` and `cnn_light`; recoverable from git history at the commit `e849d32`.
- Models renamed: `linear_material`, `linear_pst`, `linear_pst_terms`, `linear_king_bucket`, `net4_pst`.
- Effort is one million positions per point (`--effort 15` = 15 M positions, `1 + effort // 10` epochs).
- Quiescence search is on by default for the search bots (4 plies; `quiescence_plies=0` turns it off).
- Removed the Prolog pattern detector and its documentation.
- New docs layout: five public pages plus the index (`README`, `ARCHITECTURE`, `SEARCH_ALGORITHMS`,
  `ADDING_A_MODEL`, `RESULTS`); `ADDING_A_BOT`, `ROADMAP`, `TACTICAL_PATTERNS` and `docs/training/*` merged or removed.
- Per-move logs of the tournament games (`--logs`).
- New flags: `--seed` (seed of the TRAIN sample) in the training CLI and `--openings` in `run_tournament.py`.
- The agent vault (`vault/`) is no longer versioned: it stays local and is git-ignored.

### Added
- Linear evaluation models trained from the raw Lichess evaluations at an effort level 0-100
  (`training.train --effort`): `material_linear`, `pst_tapered` and `pst_terms`. The command prints
  the chosen configuration and an estimated duration before training.
- `comparison/judge.py`: Spearman, mean absolute error on the expected-score scale and best-move
  agreement with Stockfish, for all, quiet and non-quiet positions.
- Depth-accuracy profiler at a fixed time per move (`comparison/tradeoff.py`,
  `scripts/bench/profile_tradeoff.py`).
- Round-robin tournament at equal time per move with paired openings, scores and Elo
  (`comparison/tournament.py`, `scripts/bench/run_tournament.py`).
- Raw data pipeline: Lichess dump converted to sharded Parquet (`scripts/data/fetch_lichess_raw.py`,
  `training/lichess_raw.py`), sidecars with split and quiet flag (`scripts/data/prepare_dataset.py`,
  `training/sidecar.py`), and a seeded sampler.
- `scripts/bench/benchmark_search.py`: fixed-depth search benchmark that checks the chosen moves and node
  counts between two runs.
- Model cards carry accuracy, cost, trade-off and tournament sections; `MODELS_HISTORY.md` is
  generated from them.
- `tests/test_docs.py` checks the relative links, the `python -m chess_engine.*` commands and the
  `scripts/` paths quoted in the documentation; the slow tests are marked `slow`.
- Search speed-ups: fast transposition key and make/unmake in the advanced search, fast terminal
  detection with the same draw rules.

### Removed
- `RandomBot`, `HeuristicAlphaBetaBot`, `MLBot`, `TacticalAlphaBetaBot`, `NonTacticalAlphaBetaBot`
  and the `FixedDepthBot` base. `PrologPatternDetector` moved to `knowledge/pattern_detector.py`.
- `linear_baseline`, superseded by `material_linear`.
- `scripts/plot_models.py` (it read the legacy benchmark section of the cards).
- The strength matches in `comparison.evaluate` and the "Playing strength" column: playing strength
  now comes from the round-robin tournament.

### Changed
- `scripts/run_all.sh` runs the whole flow: sidecars, the linear models at `EFFORT`, evaluation of
  every model, depth profile, tournament, models table.
- `comparison.evaluate` judges every model on a fixed test sample (200,000 positions of the raw
  shards); the CSV benchmark stays available with `--data`.
- Code moved to an installable package in `src/chess_engine/` (`adversarial_search` is now
  `chess_engine.search`); module commands are `python -m chess_engine.training.train` and
  `python -m chess_engine.comparison.{evaluate,report}`.
  `ROADMAP.md` moved to `docs/`, `TRAINING_TASKS.md` and `MODELS_HISTORY.md` to `docs/training/`.
- Comparison tools (`judge`, `tradeoff`, `tournament`, `evaluate`, `report`, `metrics`) moved from
  `training` to `chess_engine.comparison`; the model cards module moved to `chess_engine.models.cards`.
  `training` no longer depends on `comparison`.
- Scripts are grouped by purpose: `scripts/data/`, `scripts/bench/`, `scripts/docs/`; `run_all.sh`
  stays in `scripts/`.
- `AdvancedHeuristicAlphaBetaBot` is now `IterativeHeuristicBot` and `AdvancedMLBot` is now
  `IterativeModelBookBot`. Strength matches play against the classical heuristic only.

## 1.0.0

First public version, derived from four chained university practicals (adversarial search,
logical reasoning, machine learning, optional extension) and then reworked.

### Added
- `models/` and `training/`: a registry of evaluation models plus train / evaluate commands.
- `evaluation/` package (hand-crafted features and heuristics), `knowledge/opening_book.py`.
- Shared bot bases (`FixedDepthBot`, `IterativeDeepeningBot`), `TimeManager`, bounded
  transposition table, `get_bot(name, **options)` in the registry, `--model` / `--depth` in the CLI.
- Scripts to build datasets (Lichess, Stockfish), plot results and render the pattern diagrams.
- 190+ tests, CI, lint and formatting configuration.

### Fixed
- Search: faster wins/slower losses were mis-weighted, an interrupted iterative-deepening iteration
  could replace a finished one, search statistics and time limits restarted at every iteration.
- Bots: `HeuristicAlphaBetaBot` and `MLBot` returned unusable values, `MLBot` never searched,
  the capture-ordering callback was never used, the opening book was disabled after the first game,
  transposition tables leaked between games.
- ML evaluation: centipawn-to-score scaling squeezed all evaluations into a tiny band, and the
  signs of the shipped CNN weights were inverted.

- Game manager: verbose output always showed UCI instead of algebraic notation, and games lost on
  time, by an illegal move or by an exception were exported to PGN with result `*`.

### Changed
- Tactical bot restored as an optional bot (needs SWI-Prolog).
- Bots find shared modules through an underscore convention (`bots/_*.py` are not registered).
- Documentation and comments are now in English.
