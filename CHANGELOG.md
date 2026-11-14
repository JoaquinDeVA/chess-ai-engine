# Changelog

All notable changes. The newest are first.

## Unreleased

### Fixed
- Endgame tables no longer loop or hang material: a winning capture, pawn move or promotion is now
  ranked first (DTZ 0) instead of by the DTZ of the position it leaves. A live game had repeated
  `Qb2`/`Qa1` instead of taking a free pawn and gave up a queen with `Qc7+ Kxc7`. Covered by two
  tests on the positions of that game.

### Changed
- Reverse futility pruning now prunes from 60 centipawns per ply left instead of 80. `uci ttd 9 rust/bench/middlegame.epd 30`: 7,099,976 to 6,450,756 nodes (−9.1 %); SPRT `rfp_margin_60` (10+0.2) queued. Raising the depth limit to 7 left the tree unchanged at depth 9 (the margin of 560 is rarely reached), so it was not tested.
- The null-move reduction now grows with the depth (2 plies from depth 3, one more every four) and with how far the static value is beyond the bound (one ply per 200 centipawns, two at most), instead of 2 or 3 by depth alone. `uci ttd 9 rust/bench/middlegame.epd 30`: 7,416,608 to 7,099,976 nodes (−4.3 %); SPRT `null_r_by_eval` (10+0.2) queued.
- Endgame tables no longer choose the move: they restrict the root of the search to the moves that
  keep the best result and, in a win, have the lowest DTZ; a single move is played without
  searching. In 100 self-play wins from random positions (three seeds, 20 ms per move, against the
  tables) the engine mates in all 100 with no repetition, and the live game's `Qc7+ Kxc7` is no
  longer an option. Some sacrifices that the DTZ counts as progress remain.
- Check positions now receive one extra search ply, and quiescence searches all legal evasions
  without stand pat or SEE when the side to move is in check. No strength or tree measurement has
  been run yet.
- Killer moves are now indexed by search ply instead of remaining depth, so IIR, LMR and null-move
  reductions reuse the same killer slot for the same layer of the tree. No strength or tree
  measurement has been run yet.
- Late move reductions now also probe eligible late quiet moves in null-window nodes; the full-window
  re-search remains part of PVS only when the parent has a full window. The search-technique module
  and its reduction names now describe their responsibilities explicitly; no strength or tree
  measurement has been run yet.
- `lmr`, `rfp`, `iir` and `futility` are a fixed part of the search, like `incremental`, `tt_move`, `null_move` and
  `see`: eight techniques, none optional. `uci bench 5`: 139,252 to 106,583 nodes (−23.5 %), median 0.115 s to
  0.091 s. `futility` enters without a verdict; the SPRTs of the four at 60 s + 1 s keep running only to confirm
  ([Results](docs/RESULTS.md#search-techniques-at-60-s--1-s)).
- New time management in `uci`. Each move gets a base of the time left over 20 moves (or `movestogo`) plus three
  quarters of the increment, minus 50 ms for the GUI; a soft limit of 0.6 times the base, checked between iterations
  and scaled by how long the best move has held and by the share of the nodes it took (×0.51 to ×5.1, the hard limit
  still caps it), and a hard limit of three times the base, at most 40 % of the time left. Before, a move took 15 % of
  the time left (9 s on the first move of 60 + 1) and the increment was ignored. The rule that predicted the next
  iteration is gone. Against the old manager (SPRT [0, 5] at 60 + 1, `data/results/matches/tm_vs_legacy/`) it scored 54.56 % in 1,020 games, +31.8 ± 17.3 Elo, LLR +1.53 of ±2.94: stopped by hand without a verdict, with no losses on time.
- The clock budget spreads the time over `max(20, 50 − move/2)` moves instead of 20, the soft limit is 0.4 of the
  base instead of 0.6, and an iteration is not started when 2.5 times the last one would no longer fit under the hard
  limit, so the hard limit no longer throws one away. In one 300 + 3 game the engine still has 173 s on the clock at
  move 30 (49 s before) and the cut iterations fall from 8.5 % of the time to 1 search in 58. One game, not an Elo
  measurement: no SPRT has been run for this change.
- `chess_lab` is organised by stage, `models` → `datasets` → `training` → `measure`. The new `datasets` package holds the
  raw shards, sidecars and split with their six commands, all run as `python -m chess_lab.datasets.<name>`. Fitting
  (`epochs`, `neural`) moves to `training`, `export_uci_model` becomes `chess_lab.models.export_uci`, the round robin and
  its `queue.json` move to `measure`, `run_full_pipeline.sh` becomes `src/chess_lab/pipeline.sh` and `capturas_gui`
  becomes `chess_lab.screenshots`. `scripts/` is gone.
- Trained model files (`.npz`) leave git for `data/models/`; their cards stay in git as the record.
- The scripts that drive the compute node leave the repo; the retired `match.sh`, `deploy.sh` and `env.sh` are deleted.
- The search scores in whole centipawns of the logistic scale (`round(logit × 173.7)`, within ±798 for a network)
  and a mate as 31,000 minus its distance in plies, so a quicker mate is worth more without the old `0.0001` per ply.
  The `rfp` and `futility` margins are 80 and 120 per ply left (the 0.46 and 0.69 log-odds of before) (`345c3ff`).
- The network runs in `f32` with a stack of accumulators (a move copies the top frame and updates it, undoing pops
  it), and one lock-free transposition table serves one thread or many. `uci ttd 11` (one thread, desktop): median
  5.06 s to 4.19 s, 919,477 to 1,268,761 nodes/s. The tree changes (`uci bench 5`: 145,163 to 139,252 nodes); the
  SPRT [−5, 0] against 4.0.0 is queued.
- `rust/` has one module per concept: network, accumulator, pruning, quiescence, bench (`15eb630`).
- `uci bench` clears the memory outside its timer, so that the first touch of a fresh table is not counted.
- `python -m chess_lab.models.export_uci` exports `net_pst_w64@e30` by default, the network the binary embeds.
- A round robin job can give engine B its own binary (`uci_b` in `queue.json`).

### Removed
- `lmp`, `aspiration`, `history_malus` and `qs_tt`: each one added to `lmr,rfp,iir` reached no verdict at 60 s + 1 s
  (LLR −0.48, 0.01, 0.07 and −0.06 of ±2.94 after 600, 600, 236 and 200 games) and the SPRTs were stopped.
- The switches: the UCI option `Techniques`, `TTD_TECHNIQUES` of `uci ttd`, `_core.TECHNIQUES` and
  `_core.DEFAULT_TECHNIQUES`, and the technique boxes of the settings dialog. A change is now measured by playing two
  binaries (`uci_b` in `queue.json`).
- The `h1` heuristic, from the core and from the laboratory (`fee2c47`).
- `TTD_TABLE` and `TTD_ENTRIES` of `uci ttd`: there is a single table.

## 4.0.0 (2026-11-11)

### Added
- `uci ttd <depth> <positions> [limit_s]`, the time to a fixed depth over the 20 positions of `rust/bench/middlegame.epd`, with counters of the search (quiescence, table, cuts by killer, repetitions); the tree is unchanged (`uci bench`: 349,598 nodes before and after). See [Measurement](docs/MEASUREMENT.md#time-to-a-depth).

### Changed
- **Documentation reduced to four files**: `README.md` is now a front page (what it found, quick start, window screenshots, the path of one move, where the data comes from, commands), and `docs/` keeps only `RESULTS.md` and `MEASUREMENT.md`. The search, techniques, models, data, training, window, architecture and openings pages moved into the private study guide.
- `src/chess_lab/scripts/capturas_gui.py` saves the window screenshots of the README offscreen (`docs/img/gui-*.png`).
- The CI test that compared tuples with `pytest.approx` compares indices and values apart, and the workflow actions run on Node 24.
- The `iir` line was repeated in every release of this file; it now appears once.
- Internal iterative reduction (`iir`) is on by default: a node of 4 plies or more with no table move is searched one ply shallower. `uci ttd 11` (1 thread): 215,120,644 to 108,502,704 nodes (-49.6 %), median 8.22 s to 4.15 s; SPRT queued. 
- Futility pruning (`futility`) and late move pruning (`lmp`) are implemented as switches, **off by default** until their SPRT: `uci ttd 11` nodes 215,120,644 to 144,499,072 (-32.8 %) and to 66,027,302 (-69.3 %). See Futility pruning and Late move pruning.
- Reverse futility pruning (`rfp`) is on by default: at 6 plies or fewer, a node whose static value is beyond the bound by 0.46 log-odds per ply left (not in check, bounds outside the mate range) returns that value. `uci ttd 11` with `lmr` (1 thread): 324,595,891 to 215,120,644 nodes (-33.7 %), median 11.13 s to 8.58 s; SPRT queued. See Reverse futility pruning.
- Late move reductions (`lmr`) are on by default: the quiet moves from the fourth on are searched shallower first and again at full depth if they improve. `uci ttd 11` (1 thread): 1,004,922,392 to 324,595,891 nodes (-68 %), median 23.53 s to 10.82 s; screening 51.33 % in 300 games, SPRT queued. See Late move reductions.
- The quiescence search tries the captures and promotions by gain (victim plus promotion minus attacker) instead of in generator order: `uci ttd 9` nodes 113,050,947 to 107,372,423 (-5.0 %), same value at a fixed depth (`uci bench`: 307,907 nodes). See Quiescence search.
- The transposition table is keyed by the position alone, without the two move counters: the same position reached by two move orders now shares its entry. Mates are stored by their distance from the node, an entry is not used or stored when the fifty-move clock could change its value, and a position with a repetition behind it skips the table. `uci ttd 9` nodes: 122,806,437 to 117,358,262 (-4.4 %), and 113,050,947 with the table at 2,000,000 entries instead of 500,000 (`uci bench` is now 340,880 nodes); the Elo effect is not measured (SPRT queued). See Transposition table.
- The preference for quicker wins and slower losses is `0.0001` per ply instead of `0.003` (a mate in 9 is worth `0.9991`, not `0.973`), so it never outweighs a real difference between two lines. The Elo effect is not measured (SPRT queued).
- The search is 21 % faster per node with the same tree (`uci bench`: 349,598 nodes; 1,088,861 to 1,316,298 nodes/s, median of 5 alternating pairs, desktop): the clock is read every 2,048 nodes, the moves are ordered on the stack and the Zobrist key follows the move (Rust core). The Elo effect is not measured.

## 3.7.0 (2026-11-10)

### Added
- Three networks trained on the GPU: `net_pst_w64`, `net_pst_w64x16` and `net_pst_k4_w64` (`59a33dd`). Every model was timed again on the same laptop.
- `entrenar.sh` resumes only the models named in `SOLO` (`d2c029b`).

### Changed
- The round robin runs with 6 concurrent games (`0bc7b69`).
- The node chains the work after training and queues the 10+5 match alone (`6d15f3a`).
- The SPRTs played without the incremental accumulator are dropped and the matched pairs are queued (`c564f5b`).

## 3.6.0 (2026-11-09)

A switch to the network: the engine plays `net_pst_w32` by default, the logistic models leave the code, and the book and the tablebases move into the UCI binary.

### Removed
- The switches `aspiration`, `history_malus` and `qs_tt`, with their UCI values and tests (`efb595e`): in the round robin at 10 s + 0.1 s none reached a verdict in about 1,100 rounds (removing `aspiration` +10.7 ± 12.5; adding `history_malus` +9.2 ± 12.0, `qs_tt` −0.3 ± 12.5). The search has four switches: `incremental`, `tt_move`, `null_move`, `see`. Each page keeps its results as "retired, to be taken up again" (Search techniques).
- The logistic models (`linear_pst_terms`, `linear_pst`) from the core and the laboratory, and the `Kings` feature set (`8700e6f`); the last commit that plays them is `a1bdf60`. Its card stays, with status `retired` (Models).

### Added
- `pvs` and `history` are part of the core, with no switch (`70701f7`). Their SPRTs [−5, 0] (`abl_pvs`, `abl_history`) continue in the round robin from LLR +2.21 and +2.09.
- The `pst_k4` feature set (piece squares in 4 king buckets, without the classical terms) and its network, King-bucket piece-square network (`net_pst_k4_w64`), registered and not trained (Models).
- Round robin of SPRTs on the compute node (fastchess, 10 s + 0.1 s, 2026-11-06 to 2026-11-09): the removal of each default technique and the models in steps; 17 jobs in [Results](docs/RESULTS.md#round-robin-rust-engine-fastchess). Removing `null_move` (−97.4 ± 24.5), `see`, `tt_move` or `incremental` loses; `net_pst_w32` beats `linear_pst_terms@e30` by +165.1 ± 32.9; the hybrid networks, without an incremental accumulator, lose to it.
- The worker installs each engine build in its own path (`bin/<hash>/uci`) and each queue job names its binary, so a long SPRT keeps its build (`a1bdf60`).
- The training task `entrenar` sets up and trains the candidate networks on the node (`91f807f`).

### Changed
- The search lists the legal moves in shakmaty's own order instead of sorting them into python-chess' order (`ordered_moves` in `rust/src/board.rs`). On one laptop at fixed depth 6 (`uci bench`) it ran about 28 % more nodes per second and about 8.7 % more nodes (estimated, one run, no SPRT); the Elo effect is not measured. Equal-value ties can resolve to another move (the K+Q vs K tablebase test now expects `a7b6`).
- The `uci` binary consults the opening book and the endgame tablebases itself, through the UCI options `Book` (JSON file) and `Tablebase` (folder), off by default so the matches and tournaments are unchanged; the window sets them. The Python wrappers (`chess_engine/knowledge/`, `_core.Book`, `_core.Tablebase`) are gone; `position_key` lives in `build_opening_book.py` (Opening book, Endgame tablebases).
- Every network has an incremental accumulator, not only the slotted piece-square ones: the bucketed and hybrid networks update their piece part move by move, rebuild it when the king changes bucket, and add the classical terms at each leaf. `uci bench 5`: `net_hybrid_w64` 521,915 to 635,828 nodes/s (+22 %), `net_hybrid_k4_w64` +16 %; about 2 % less for `net_pst_w32` (incremental).
- The UCI binary plays the piece-square network `net_pst_w32` (embedded, `rust/models/net_pst_w32.json`) instead of `linear_pst_terms` (`8700e6f`): 1,020,805 nodes/s in `uci bench 5` (laptop). `setoption name Model` loads networks only. Training code shared with the networks moved to `models/fitting/epochs.py`.
- Documentation centred on the network (Models); the retired techniques and the logistic model keep a section with what was measured.
- The opening suite for the matches has 10,000 openings (`107b9e7`).

## 3.5.0 (2026-11-07)

### Added
- A round-robin scheduler for SPRT matches on the compute node (`4654821`); the node starts it, uploads the engine and collects the results (`35387f0`, `3d8c467`). A stop task ends it (`3147519`).
- The `uci` binary loads networks and linear models from a JSON file (`b531f9a`).
- The models are compared in steps instead of all against the linear one, and the best networks also play the linear model directly (`88c31c3`, `d2d03ba`).

## 3.4.0 (2026-11-05)

### Added
- Trained hybrid networks (768 piece-square inputs plus the eleven classical terms, optionally with 4 or 8 king buckets) at effort 30: Spearman 0.702 to 0.722 against 0.646 for the logistic model, 3.4 to 3.9 µs per evaluation against 2.4, all measured on one laptop (`python -m chess_lab.measure.evaluate`). Not played yet. See Models.

### Changed
- `chess_lab` reorganised: `models/features.py` (was `linear_features.py`), `models/target.py` (was `representation.py`), `models/fitting/{linear,neural}.py` (were `linear_eval.py`, `neural_eval.py`), `training/data/{raw_shards,sidecar,split}.py` (were `lichess_raw.py`, `sidecar.py`, `dataset.py`) `training/cache.py` (cache directory, split from `effort.py`) and `models/store/` (`artifacts/`, `cards/`, `h1_weights.json`, `loader.py`, `cards.py`). The Lichess data directory is now `data/LichessPositions/` and the encoded-batch cache sits directly in `data/cache/`. Same behaviour; imports and paths in the docs follow.
- Every trained network is measured on one laptop with `python -m chess_lab.measure.evaluate` and listed by name (`63427fb`).

## 3.3.0 (2026-11-04)

### Added
- A move-ordering switch, off by default and measured at a fixed 20,000 nodes: `history_malus` (quiet moves tried before a cut lose history score; +3.9 ± 9.3 Elo, no verdict). Results in [RESULTS](docs/RESULTS.md).
- The `--patience` option of the training: stop after N epochs without improvement (`33497b7`).
- The king-bucket network `net_kings_w32` and its feature set `pst_kings` (`57a23aa`).
- Several threads: each helper runs its own iterative deepening with a skip pattern and aspiration, the history scores are shared, the table slots are cache-aligned and the move is chosen by a vote of the threads. Seconds to depth 8 with 4 threads: 0.788 → 0.638; with 6: 0.871 → 0.627 (`python -m chess_lab.measure.search_speed`). In games 6 threads against 4 gave +5.0 ± 37.5 Elo in 280 games at 10 s + 0.1 s (no verdict), so 4 is the recommended count. The vote itself, at 6 threads against the same code without it: −36.0 ± 37.9 Elo in 300 games (`data/results/matches/smp_vote/match.log`), not significant.

### Changed
- Static exchange evaluation, `see`, now **on by default** (binary and window): SPRT [0, 5] at a fixed 20,000 nodes per move accepted H1, +56.4 ± 17.5 Elo in 1,188 games (`data/results/matches/see_nodes/match.log`). At a time control it had +47.8 ± 34.3 in a 300-game screening and +41.5 ± 46.1 and +67.2 ± 34.5 in two SPRTs stopped by hand (no verdict).
- Two more speed-ups of the Rust search, same nodes: the build targets x86-64-v3 (AVX2, BMI2; `.cargo/config.toml`), 750 k to 788 k nodes/s (+5 %), and `linear_pst_terms` follows its piece-square sums move by move (`incremental`), 788 k to 928 k (+17 %), 275,144 nodes in `uci bench` (median of 5 runs, laptop). Against the binary before the 2.6x speed-up, same techniques without `see`: 73.83 % in 300 games at 10 s + 0.1 s, +180.2 ± 36.0 Elo at the last rating printed (280 games; `data/results/matches/speed_vs_old/match.log`).
- The Rust search is 2.58x faster in `uci bench` (304 k to 784 k nodes/s, the same 565,744 nodes; 2.2x at depth 8 on 60 other positions, 302 k to 661 k, with a one-off harness that was not kept): packed move-order keys, pawn terms on bitboards, no allocation per node in the logistic model, one list in move ordering. +130.9 ± 39.8 Elo against the previous binary in 300 games at 10 s + 0.1 s. The Rust modules were renamed after what they hold (see Architecture).
- Documentation reduced to what is current: the models page covers `linear_pst_terms`, `h1` and the light network candidates; figures of the earlier Python engine and of removed models are gone or marked "not measured"; the techniques index has an "Elo estimated" column.
- Training uses quiet positions always; the `Representation` filter option is gone (`30d4a85`). Models and features that were discarded are removed (`29549a9`).

### Removed
- The switches `lmr`, `qs_evasions` and the UCI option `TimeRules`, all measured and rejected (Search techniques).

## 3.2.0 (2026-11-03)

### Added
- A quiescence switch, off by default and measured at a fixed 20,000 nodes: `qs_tt` (the value of a horizon node goes into the table; +0.7 ± 3.9 Elo, no verdict). Results in [Results](docs/RESULTS.md).
- The quiescence switch `qs_evasions` and the UCI option `TimeRules` (single, half, stab, nodes), both off by default and rejected later (`e2cea5b`, `30cba50`).
- Passed-pawn and king-attack term models, `linear_pst_passed` and `linear_pst_king` (`2e848cb`), and the `see` technique, off by default, to be measured (`6caf4dc`).

### Changed
- The shipped `linear_pst_terms` is now trained on quiet positions only (effort 10): +284 ± 43 Elo against the previous model in 362 games at 60 s + 1 s. Three variants (`k` 0.0065, 0.5 % of the mates, 30 M positions) did not beat it. See Models and [Results](docs/RESULTS.md).
- The clock of the UCI binary is again the share of the time left (15 %, between 1.5 and 40 s): the rule that spent the time left over the moves to go plus 75 % of the increment lost −64 ± 24 Elo at 60 s + 1 s in 720 games (SPRT stopped by hand), see [Results](docs/RESULTS.md).
- [How a change is measured](docs/MEASUREMENT.md) says which time control to use and that fixed nodes favour changes that save nodes. The worker notes in `match.sh` describe the port forwarding of the desktop's SSH.

## 3.1.0 (2026-11-01)

A cleanup of 3.0: every piece serves playing or the laboratory, the models live in the laboratory, and the
laboratory judges every model with the core's evaluator. Parts of the `_core` API and some laboratory names
changed (below).

### Added
- **Reader-first documentation**: a front page with reader paths, one page per search technique (problem, idea, how it works with a worked example, implementation, measured effect, status) with light and dark diagrams, How the engine chooses a move and [How a change is measured](docs/MEASUREMENT.md) (fastchess, SPRT, the A/A check). [Results](docs/RESULTS.md) separates what is measured from what is estimated and marks the Python-engine tournaments as history.

### Changed
- The extra `ml` no longer installs `matplotlib` (nothing imports it). The worker scripts read the worker's address and the repository root from `src/chess_lab/scripts/worker/env.sh`; `keepalive.sh`, which managed the laptop's wifi, left the repository.
- The results table holds what can be measured again today (accuracy and cost): the depth, search agreement and tournament columns of the Python engine, and the `tradeoff` and `tournament` card sections they came from, were removed; those figures stay in the prose of [Results](docs/RESULTS.md).
- The data directories and the split seed live in `chess_lab.models.defaults` (`RAW_DIR`, `SIDECAR_DIR`, `SPLIT_SEED`), absolute from the repository root, so the commands find the data from any working directory. `run_full_pipeline.sh` trains every registered model by default (it asks the registry instead of keeping its own list).
- Renamed: `build_opening_book.white_scores` to `ranking_scores`, `lichess_raw.material_difference` to `probe_material_balance`, `linear_eval._losses`/`_scores` to `losses`/`scores`. `build_openings.py` moved to `src/chess_lab/scripts/data/`.
- The laboratory judges and times every model with the core's `Evaluator`, as the engine plays it: `judge`, `evaluate`, `check_split_leakage` and the effort estimate no longer evaluate in Python (`PythonEvaluator` and `HeuristicModel` were removed). The cost column of [Results](docs/RESULTS.md) is now the core's for every model, from a FEN, and scores are clipped to 0.01–0.99 like in play; correlations and errors are unchanged to 4 decimals; best-move agreement moved by at most three positions per model (at most 0.0005, in `linear_pst`'s non-quiet positions).
- The models left the application: `chess_engine.models` is now part of `chess_lab.models` (the loader, `h1_weights.json` and `artifacts/` are under `src/chess_lab/models/`). `load_evaluator` moved to `chess_lab.models.definitions`, and the export to the UCI binary reads models through `load_model`. The repository root is `chess_engine.paths.ROOT`.

### Removed
- From `chess_engine._core`: `Searcher` (the search runs only in the UCI binary), `estimate`, `version` and the constants `DEFAULT_ESTIMATOR`, `DEFAULT_MIN_DEPTH`, `DEFAULT_QUIESCENCE_PLIES`, `DEFAULT_MAX_TT_ENTRIES` and `ESTIMATORS`. `_core.DEFAULT_TECHNIQUES` and `_core.MAX_DEPTH` are new, and the window reads them instead of its own copies. The time estimators `ratio_squared`, `clamped_growth`, `median_growth` and `node_growth` and the search switches that turned off the table, the killer moves and the capture ordering are gone; the moves and node counts are unchanged (`uci bench`).
- The Decisions, Pitfalls and Remote worker pages: what was decided is in the page of each technique or model.
- The page on tactical move ordering with Prolog: the code went in `ca35218` (2026-10-18) after costing about 2 ms per node; it was never measured in games.

## 3.0.0 (2026-10-31)

A change of focus: the project becomes an engine to play against in a window, and the Rust engine speaks UCI.
The application plays through the UCI binary; the laboratory is its own package and plays matches with
fastchess instead of its own tournaments. The console game, the Python bots and the Keras course models are gone.

### Added
- **A window to play** (`python main.py`, `chess_engine.gui`, optional extra `gui` with PySide6): board in the chess.com style, pieces by cburnett (GPLv2+, `src/chess_engine/gui/assets/pieces/LICENSE.md`), human vs AI and AI vs AI, clocks, move list with navigation, undo, draw, resign, and a Settings dialog per AI (depth, time per move, opening book, tablebases, search techniques, threads). See The window.
- **A UCI engine**, `rust/target/release/uci` (`cargo build --release --bin uci`): the search of the Rust core behind the UCI protocol (`position`, `go depth|movetime|wtime … winc … binc|nodes|infinite`, `setoption` for `Threads`, `Hash`, `Techniques` and `Model`, `stop`, `bestmove`). The window plays through it with `chess_engine/uci.py`. See Architecture.
- **An opening suite for matches**: `src/chess_lab/scripts/build_openings.py` writes balanced positions (8 to 10 half-moves from the start, Stockfish within 0.5 pawns) for fastchess. See Openings.
- **A remote worker for matches**: `src/chess_lab/scripts/worker/` deploys the UCI binary and fastchess to the desktop and plays matches there over SSH.

### Changed
- The window's AI is the UCI engine: the Python bot classes, the Python game loop and the Python search wrapper are gone. The window keeps the book and the tablebases in Python, before the engine. The engine has one evaluation (`linear_pst_terms`) and quiescence fixed at 4 plies, so the Settings dialog no longer offers them.
- The laboratory measures models with `python -m chess_lab.measure.{report,evaluate}`. Games between engines are no longer played by this repository: fastchess plays them against the UCI binary. The pipeline `run_full_pipeline.sh` no longer runs tournaments.
- Dependencies: `numpy` is a base dependency; `joblib` and `pytest-cov` were dropped.
- Search depth is decided by the clock: the bots no longer default to `max_depth=3` and `min_depth` is 1 (was 3); `max_depth` is an optional explicit limit (`None` = no limit, technical cap `MAX_DEPTH` = 30). In the window, "Maximum depth" 0 means no limit.
- Training moved out of the application: `chess_lab.models` now holds the feature extractors, `representation.py`, `spec.py`, `new_model`/`fit_stream` and `PythonEvaluator`. `chess_engine.models` only loads, saves and registers models (`LinearEvalModel(weights, bias, rep)` takes the weights instead of building zeros; `rep` is a plain dictionary and the `.npz` format is unchanged). `EvaluationModel` lost `predict_cp`, `predict_cp_batch` and `cp_to_expected_score`; only the laboratory evaluates in Python.
- The laboratory is its own package, `chess_lab` (`src/chess_lab/`): `training/`, `comparison/` and the former top-level `scripts/` left `chess_engine`. Commands change from `python -m chess_engine.training.train` and `chess_engine.comparison.*` to `python -m chess_lab.training.train` and `chess_lab.comparison.*`, and scripts from `scripts/...` to `src/chess_lab/scripts/...`. The application no longer imports the laboratory.

### Removed
- The reference files that pinned the core to the Python engine (their test and the Rust tests that read them); the rules are checked by perft and end-of-game tests in `rust/tests/rules.rs`, the book and tables by `rust/tests/knowledge.rs`, and the positions of the technique tests moved to `tests/data/positions.json`.
- The Keras course models (`mlp_features`, `cnn_resnet`) and the "Kind" column of the results table; the Python evaluation fallback (`evaluation/`, `ChessGameState`, `ModelEvaluator`, `callback_evaluator`): the core evaluates every bot, so the Keras dependency is only for training `net4_pst`. `h1_weights.json` moved to `models/` and stays fixed (`fit_h1_weights.py` and the Python feature code were removed).
- `chess_engine.config` (`BLITZ_CONFIG`, `load_config`) and `TournamentConfig`, which nothing used.
- The Python game loop and bots (`chess_engine.bots`, `chess_engine.game`), the Python tournament (`chess_lab.tournaments`, `chess_lab.players`, `measure/tradeoff.py`) and the experiment scripts in `scripts/bench/`. The results they produced stay in [Results](docs/RESULTS.md); their code is in the git history before this version.
- The console game: the ~17 options of `main.py`, `HumanPlayer`, the board printing of `GameManager` and `--mode`; `main.py` now only opens the window.

## 2.1.0 (2026-10-30)

Fixes after 2.0.

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
  replaces, only faster. Installing needs a Rust
  toolchain; CI runs `cargo fmt`, `clippy` and `cargo test`. See The search in Rust.
- **Six search techniques** behind switches (`--techniques LIST|all|none`, constructor option `techniques`): incremental
  accumulator for `net4_pst`, table move first, history heuristic, principal variation search, aspiration windows
  and null move; late move reductions (`lmr`) are implemented but off because they lost games. The bots use all but
  `lmr`: 76.5 % against the reference search in 100 games at equal time (+205 Elo, one model). Measurements and
  tournaments (`make_player`) keep the reference search unless the player is written `model+tech1,tech2`. See
  Search techniques.
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
  engine is recoverable from git history.
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
  `data/knowledge/endgame_tablebase` exists) unless told otherwise (`--no-opening-book`, `--no-endgame-db`, or `None` for
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
