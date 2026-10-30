# Changelog

All notable changes. The newest are first.

## Unreleased

### Changed
- The tournament script moved from `scripts/bench/play_tournament.py` to `chess_engine.comparison.play_tournament`; it takes the same options and runs with `python -m chess_engine.comparison.play_tournament`.

### Fixed
- `_core.TECHNIQUES`, and with it `--techniques all` and the `+all` suffix of the tournament players, now include `incremental`; before, `all` left it out.
- `get_info()` of the model bot works without an opening book (`book_positions` is 0).
- The PGN `Event` header is "ChessAIEngine game".

### Removed
- `TimeManager.expired` and `game_manager.load_config`, which nothing used.

## 2.0.0 (2026-10-26)

The search moves to Rust. The engine plays the same moves as 1.1 with the reference search, 8 to 35 times faster,
and adds six selective-search techniques and several threads. Installing needs a Rust toolchain. Tournament
results published for 1.1 were played with the Python engine and are not comparable with 2.0.

### Added
- **The search, the evaluations and the knowledge in Rust** (`rust/`, the Python module `chess_engine._core`, built
  with `maturin`): the rules (on `shakmaty`), alpha-beta with iterative deepening, transposition table, killers,
  MVV-LVA and quiescence, the time estimators, the evaluations of `h1`, the four logistic models and `net4_pst`, the
  opening book and the Syzygy tables. It plays the same moves with the same node counts as the Python engine it
  replaces (`tests/golden/`, `tests/test_core_golden.py`, `rust/tests/`), only faster. Installing needs a Rust
  toolchain; CI runs `cargo fmt`, `clippy` and `cargo test`. See [Algorithms](docs/algorithms/rust-and-threads.md).
- **Six search techniques** behind switches (`--techniques LIST|all|none`, constructor option `techniques`): incremental
  accumulator for `net4_pst`, table move first, history heuristic, principal variation search, aspiration windows
  and null move; late move reductions (`lmr`) are implemented but off because they lost games. The bots use all but
  `lmr`: 76.5 % against the reference search in 100 games at equal time (+205 Elo, one model). Measurements and
  tournaments (`make_player`) keep the reference search unless the player is written `model+tech1,tech2`. See
  [Algorithms](docs/algorithms/selective-techniques.md).
- `scripts/bench/compare_engine_versions.py`: two search configurations of the same model play each other.
  `tests/test_feature_parity.py`: each feature family matches its Rust version or falls back with a warning.
- **`--threads N` / `--hash MB`**: Lazy SMP, the search on several threads sharing a lock-free table. One thread by
  default; the search is not deterministic with more.
- **Seven drawings** in `docs/img/` (alpha-beta cut, path of a move, move ordering, PVS and aspiration windows, null move,
  threads with a shared table, tournament format), linked from the pages that explain them; they replace the
  Wikipedia figure of the algorithms page.
- `scripts/bench/measure_search_speed.py`: depth reached in a time budget and cost of a fixed depth, per player.

### Changed
- **`python main.py` with no arguments** starts a game of you (White) against `IterativeModelBookBot` with
  `linear_pst_terms`, 3 s per move (`--move-time`) and up to four threads. Naming one player completes the other so
  that a human faces a bot.
- **Resumed tournament files must come from the same core version**: every game record carries `core`
  (`chess_engine._core.version()`); records without it count as `python` (the engine before 2.0) and resuming with a
  different version is refused.
- `IterativeDeepeningBot` takes `threads` and `hash_mb`, no longer takes `problem`, and has no `order_moves` or
  `cutoff_test` to override; a bot says which evaluation the core runs with `native_evaluator()` (the default calls
  `evaluate` back from Python, which is slow). `scripts/bench/compare_search_features.py` switches the table, the
  killers and the ordering off through `player.disabled`.
- The abstract `GameState` / `GameProblem` of the course framework and the Python search were removed with the move
  to Rust; `ChessGameState` stays as a thin wrapper of a board for the evaluation functions.

### Removed
- The Python search (`game_algorithms`, `search_utils`, `time_estimators`, `game_problems`, `chess_problem`), its tests,
  `scripts/bench/bench_search_speed.py` (replaced by `measure_search_speed.py`) and `docs/benchmarks/`. The Python
  engine is recoverable from git history; `tests/golden/` keeps its behaviour.
- `main.py --mode tournament`, `GameManager.run_tournament`, `TournamentResult` and
  `BotRegistry.get_tournament_ready_bots`: tournaments run from `scripts/bench/play_tournament.py`. `--export` now
  only names the PGN file of the game.

## 1.1.0 (2026-10-22)

Training on the Lichess evaluation database, the model family, the opening book and the endgame tables.

### Added
- **Training on the Lichess evaluation database.** The dump becomes raw Parquet shards
  (`scripts/data/build_raw_shards.py`) with sidecars for the split, the quiet flag and the side to move
  (`scripts/data/build_sidecars.py`), and a seeded sampler. `python -m chess_engine.training.train --effort N` trains on
  N million positions (`1 + N // 10` epochs), prints the configuration and an estimated duration first, can decay the
  learning rate on a plateau (`--lr-decay`) and caches the encoded batches on disk (`--no-cache` turns it off).
- **Evaluation models.** Four logistic models over sparse features (`linear_material`, `linear_pst`, `linear_pst_terms`,
  `linear_king_bucket`) and a four-layer network (`net4_pst`); `h1`, a hand-written heuristic whose six weights are fitted
  to Stockfish (`scripts/data/fit_h1_weights.py`). Each model has a card in `models/cards/`.
- **Search.** Quiescence search (on by default, 4 plies), transposition table, killer moves, MVV-LVA move ordering,
  make/unmake on one state with fast transposition keys, fast terminal detection, and an iterative deepening that always
  completes depth 3 (`min_depth`).
- **Comparison tools.** `comparison/judge.py` (Spearman, mean absolute error and best-move agreement against Stockfish),
  `scripts/bench/profile_depth_vs_accuracy.py` (depth and agreement at a fixed time per move),
  `scripts/bench/play_tournament.py` (round robin at equal time, paired openings, Elo against an anchor, per-move logs,
  `--max-depth` for fixed-depth matches), `scripts/bench/compare_quiescence.py`, `scripts/bench/compare_search_features.py`
  (what each search technique saves), `scripts/bench/bench_search_speed.py`.
- **`scripts/run_full_pipeline.sh`**, the reproducible flow from the sidecars to the results table: resumable steps,
  `DRY_RUN=1` to preview the commands.
- **Documentation** in English: README, architecture, search algorithms, data and training, models, results and the guide to
  adding a model. `tests/test_docs.py` checks the links, the commands and the script paths they quote.
- **The extras are part of the bots.** Every search bot plays the shipped opening book and the endgame tables (when
  `data/knowledge/endgame` exists) unless told otherwise (`--no-opening-book`, `--no-endgame-db`, or `None` for
  `opening_book` / `tablebase`), next to quiescence and the time estimator, which were already on. The tournaments
  and measurement scripts build their players without book and tables, to compare evaluations alone.
  `IterativeModelBookBot` defaults to `linear_pst_terms`, which ships with the repository.
- **Opening book** rebuilt from the Lichess evaluations (121,868 positions, 24 half-moves, up to four moves per
  position) with `scripts/data/build_opening_book.py`, and `scripts/data/check_opening_book.py`, which judges a random
  sample of its moves with Stockfish (98.7 % within 50 centipawns of the best line on 2,438 entries).

### Changed
- The default training keeps 10 % of the mate positions; the 100 % variant is `--mate-keep 1.0 --tag mates100`. The default
  representation name no longer carries a `-mates10` suffix.
- A model card keeps one tournament entry per time control (`60+1`, `600+5`, `depth3`), and retraining clears the
  measurements of the previous model.
- `evaluate` also judges `h1`.
- Scripts were renamed to say what they do: `build_raw_shards.py`, `build_sidecars.py`, `fit_h1_weights.py`,
  `compare_target_representations.py`, `bench_search_speed.py`, `profile_depth_vs_accuracy.py`, `play_tournament.py`,
  `run_full_pipeline.sh`.
- Models are presented with a full name in the documentation (for example "Material logistic model") and keep the short
  identifier in the code (`linear_material`).
- `h1` replaces the heuristics `h2`, `h3` and `h4`. The models `ridge_features` and `cnn_light` were removed (recoverable
  from commit `e849d32`).
- Code lives in an installable package, `src/chess_engine/` (`adversarial_search` became `chess_engine.search`).
  `AdvancedHeuristicAlphaBetaBot` is now `IterativeHeuristicBot` and `AdvancedMLBot` is now `IterativeModelBookBot`.


### Removed
- The Prolog pattern detector and its documentation (no bot used it), the CSV training path (`--data`, `load_evaluations`,
  the CSV download and labelling scripts), `comparison/metrics.py`, `training/stockfish.py`, the TOML configurations of the
  course models, the plain `minimax_search` / `alphabeta_search` / `random_moves`, and unused helpers of the game manager,
  the bot registry and the opening book.
- The simple bots (`RandomBot`, `HeuristicAlphaBetaBot`, `MLBot`, `TacticalAlphaBetaBot`, `NonTacticalAlphaBetaBot`) and the
  `FixedDepthBot` base; the model `linear_baseline`.
- The vault (`vault/`) is no longer versioned: it stays local.

### Fixed
- The iterative deepening could stop at depth 2 with most of the time budget unused (12 % of the moves of a
  10 min + 5 s tournament): the rule that predicted the next iteration squared a ratio that reaches thousands when the
  first iteration lasts microseconds. The rule is bounded now (`geometric_growth`; the old one stays selectable as
  `ratio_squared`) and depth 3 is always completed.
- The CI failed because the tests of the course models needed `*.keras` files that are not versioned; those tests now skip
  when the files are absent.
- `--epochs` was ignored when training by effort, and the bot option `max_tt_entries` never reached the search.
- Wrong help texts in `main.py`, Ctrl-C was swallowed by the human player, and the default tournament report paths pointed to
  a folder that did not exist.

## 1.0.0 (2026-10-16)

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
