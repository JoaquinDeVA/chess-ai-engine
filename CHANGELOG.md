# Changelog

## Unreleased

### Added
- Linear evaluation models trained from the raw Lichess evaluations at an effort level 0-100
  (`training.train --effort`): `material_linear`, `pst_tapered` and `pst_terms`. The command prints
  the chosen configuration and an estimated duration before training.
- `training/judge.py`: Spearman, mean absolute error on the expected-score scale and best-move
  agreement with Stockfish, for all, quiet and non-quiet positions.
- Depth-accuracy profiler at a fixed time per move (`training/tradeoff.py`,
  `scripts/profile_tradeoff.py`).
- Round-robin tournament at equal time per move with paired openings, scores and Elo
  (`training/tournament.py`, `scripts/run_tournament.py`).
- Raw data pipeline: Lichess dump converted to sharded Parquet (`scripts/fetch_lichess_raw.py`,
  `training/lichess_raw.py`), sidecars with split and quiet flag (`scripts/prepare_dataset.py`,
  `training/sidecar.py`), and a seeded sampler.
- `scripts/benchmark_search.py`: fixed-depth search benchmark that checks the chosen moves and node
  counts between two runs.
- Model cards carry accuracy, cost, trade-off and tournament sections; `MODELS_HISTORY.md` is
  generated from them.
- Search speed-ups: fast transposition key and make/unmake in the advanced search, fast terminal
  detection with the same draw rules.

### Removed
- `RandomBot`, `HeuristicAlphaBetaBot`, `MLBot`, `TacticalAlphaBetaBot`, `NonTacticalAlphaBetaBot`
  and the `FixedDepthBot` base. `PrologPatternDetector` moved to `knowledge/pattern_detector.py`.
- `linear_baseline`, superseded by `material_linear`.
- The strength matches in `training.evaluate` and the "Playing strength" column: playing strength
  now comes from the round-robin tournament.

### Changed
- `scripts/run_all.sh` runs the whole flow: sidecars, the linear models at `EFFORT`, evaluation of
  every model, depth profile, tournament, models table.
- `training.evaluate` judges every model on a fixed test sample (200,000 positions of the raw
  shards); the CSV benchmark stays available with `--data`.
- Code moved to an installable package in `src/chess_engine/` (`adversarial_search` is now
  `chess_engine.search`); module commands are `python -m chess_engine.training.<command>`.
  `ROADMAP.md` moved to `docs/`, `TRAINING_TASKS.md` and `MODELS_HISTORY.md` to `docs/training/`.
- `AdvancedHeuristicAlphaBetaBot` is now `IterativeHeuristicBot` and `AdvancedMLBot` is now
  `IterativeModelBookBot`. Strength matches play against H2 and H3 only.

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
