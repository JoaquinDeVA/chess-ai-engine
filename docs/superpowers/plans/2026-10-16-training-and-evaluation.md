# Training and Evaluation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fast linear evaluation models (material, piece-square tables, tables + classical terms) trained on the Lichess evaluations with an effort level, a bake-off that picks the data representation, and a comparison centred on the depth-accuracy trade-off decided by a large round-robin.

**Architecture:** Raw Parquet shards (already fetched) get an aligned sidecar with the split and a quiet flag; targets are derived on the fly from `cp`/`mate` by a `Representation`, so there is one stored dataset. A shared linear-model core (sparse features -> logit -> sigmoid, numpy Adam trainer, pure-Python fast evaluation) backs the three new models. Comparison tools measure accuracy (static and of the search), depth at fixed time per move, and play a parallel round-robin.

**Tech Stack:** Python 3.12, python-chess, numpy, pyarrow, pytest, ruff. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-16-training-and-evaluation-design.md`

## Global Constraints

- Code, comments, docstrings in English; reST docstrings (`:param x:` / `:return:`); line length 100; `ruff check .` and `ruff format --check .` clean; run tools from `.venv/bin/`; `export TF_CPP_MIN_LOG_LEVEL=3`.
- Package code under `src/chess_engine/`; tests are plain pytest functions in `tests/`.
- The search consumes an expected score in `[0.01, 0.99]` for a given player; terminal values are 0 / 0.5 / 1.
- Every model converts from its own training target inside the model, with the constants saved in its artifact; training and playing cannot disagree.
- Existing models (`ridge_features`, `mlp_features`, `cnn_light`, `cnn_resnet`) are not retrained, do not play matches and are never deleted; they only get accuracy and cost metrics.
- One stored dataset: raw shards plus an aligned sidecar. No per-variant copies.
- Do not change the search algorithm (only add statistics).
- Data budget ~10 GB for Lichess data; machine 6 cores, no GPU, ~6 GB RAM free in practice: stream, never load the whole dataset.
- The full download (`data/lichess/raw/`, ~410M rows) is in progress; tasks must work on small fixtures, and runs on real data happen in the Runs section.
- Do not commit generated data under `data/`.

## Review Focus

1. Black to move with `perspective="side_to_move"`: the target, the features and the model output must all flip consistently; a mirrored position must get the complementary score (tested in Tasks 1 and 3).
2. Mate rows (`cp` null): targets must be exactly 1/0 by mover, never NaN, and `cp_linear` must not divide by a null (Task 1).
3. Positions with no legal moves or a terminal state inside the 1-ply judge: checkmate counts as a win for the mover, stalemate as 0.5 (Task 5).
4. Raw shards whose row count differs from their sidecar (interrupted prepare run): the sampler must refuse, not silently misalign (Task 2).
5. A parallel tournament where one game crashes: the run must record the game as failed and continue, not lose the other results (Task 11).

---

## Track A — foundations and the representation bake-off (sequential)

### Task 1: Representation and targets

**Files:**
- Create: `src/chess_engine/training/representation.py`
- Test: `tests/test_representation.py`

**Interfaces:**
- Produces:
  - `LICHESS_K = 0.00368208`, `CP_LINEAR_SCALE = 1267.0`
  - `@dataclass(frozen=True) class Representation: scale: str = "winprob"; k: float = LICHESS_K; perspective: str = "side_to_move"; filter: str = "all"` with `scale in {"winprob", "cp_linear"}`, `perspective in {"white", "side_to_move"}`, `filter in {"all", "quiet"}`; `name` property (e.g. `"winprob-k0.00368-stm-all"`); `to_dict()` / `from_dict(d)`.
  - `def white_score(cp: np.ndarray, mate: np.ndarray, rep: Representation) -> np.ndarray` — expected score for White in `[0, 1]` from White-POV `cp`/`mate` (nullable as NaN): mates -> 1.0 if `mate > 0` else 0.0; `winprob`: `1 / (1 + exp(-k * cp))`; `cp_linear`: `0.5 + 0.5 * clip(cp / 1267, -1, 1)`.
  - `def target(cp, mate, white_to_move: np.ndarray, rep) -> np.ndarray` — `white_score`, flipped to `1 - s` where Black moves if `rep.perspective == "side_to_move"`.
  - `def score_to_cp(score: float | np.ndarray, rep) -> ...` — inverse of `white_score` for non-mate scores (clip score to `[1e-6, 1-1e-6]`), used for `predict_cp`.
  - `def is_quiet(board: chess.Board, best_move: str | None) -> bool` — not in check and best move (if any) is neither a capture (incl. en passant) nor a promotion.

- [ ] **Step 1: Write failing tests**: `test_winprob_zero_cp_is_half` (`white_score([0], [nan])==0.5`), `test_winprob_matches_lichess_curve` (cp=300 -> `1/(1+exp(-0.00368208*300))`), `test_cp_linear_clips_at_1267` (cp=5000 -> 1.0, cp=-1267 -> 0.0), `test_mates_are_one_or_zero_without_nan` (mate=+3 -> 1.0, mate=-1 -> 0.0, both scales), `test_side_to_move_flips_for_black` (same cp, white_to_move False -> `1 - s`), `test_white_perspective_never_flips`, `test_score_to_cp_inverts_white_score` (round trip for cp in [-800, 800], both scales), `test_is_quiet` (check -> False; capture `e4d5` -> False; promotion -> False; quiet move -> True; `None` best move in quiet position -> True), `test_representation_round_trips_through_dict`.
- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_representation.py -q` — FAIL (module missing).
- [ ] **Step 3:** Implement the module (vectorized numpy; `np.where` on `isnan(mate)`).
- [ ] **Step 4:** Run the tests — PASS; ruff clean.
- [ ] **Step 5:** Commit `feat: data representation and training targets`.

### Task 2: Sidecar (split + quiet flag) and sampler

**Files:**
- Create: `src/chess_engine/training/sidecar.py`, `scripts/prepare_dataset.py`
- Test: `tests/test_sidecar.py`

**Interfaces:**
- Consumes: raw shards `data/lichess/raw/part-NNNNN.parquet` (schema `RAW_SCHEMA` in `training/lichess_raw.py`: `fen, cp, mate, depth, knodes, best_move, n_evals, n_pvs`), `assign_split` and `TRAIN/VALIDATION/TEST` from `training/dataset.py`, `is_quiet` (Task 1).
- Produces:
  - `def build_sidecar(shard: Path, out_dir: Path) -> Path` — writes `out_dir/<shard stem>.sidecar.parquet` with columns `split: int8`, `quiet: bool`, `white_to_move: bool`, same row order and count as the shard; metadata `{"rows": n, "source": shard.name}`.
  - `scripts/prepare_dataset.py --raw data/lichess/raw --out data/lichess/sidecar --jobs 5`: builds missing sidecars in parallel (one shard per worker), skips existing ones, prints progress.
  - `def load_sample(raw_dir: Path, sidecar_dir: Path, split: int, n: int, seed: int = 42, quiet_only: bool = False) -> pd.DataFrame` — columns `fen, cp, mate, best_move, quiet, white_to_move`; uniform over all shards (sample row indices per shard proportionally to its size, read only the needed row groups); deterministic for a seed; raises `ValueError` if a shard and its sidecar disagree on row count or a sidecar is missing.
  - `def iter_batches(raw_dir, sidecar_dir, split, batch_size, n: int | None, seed, quiet_only=False) -> Iterator[pd.DataFrame]` — streaming version for training (shuffled shard order, shuffled within row groups), stops after `n` rows if given.

- [ ] **Step 1: Write failing tests** with a tmp dir holding two tiny raw shards written via `training.lichess_raw.write_shard`: `test_sidecar_aligns_with_shard` (row count, `split` equals `assign_split` of the FENs, `quiet`/`white_to_move` correct on hand-picked FENs), `test_load_sample_is_deterministic_and_respects_split`, `test_load_sample_quiet_only`, `test_misaligned_sidecar_raises` (truncate a sidecar -> `ValueError`), `test_iter_batches_yields_n_rows_in_total`.
- [ ] **Step 2:** Run — FAIL.
- [ ] **Step 3:** Implement (pyarrow; `chess.Board(fen)` per row for `quiet`; parallel script via `concurrent.futures.ProcessPoolExecutor`).
- [ ] **Step 4:** Run — PASS; ruff clean.
- [ ] **Step 5:** Commit `feat: dataset sidecar with split and quiet flag, sampler`.

### Task 3: Linear model core, PST features, scale contract

**Files:**
- Create: `src/chess_engine/models/linear_eval.py`, `src/chess_engine/models/linear_features.py`
- Modify: `src/chess_engine/models/base.py` (add `expected_score`), `src/chess_engine/models/evaluator.py`, `src/chess_engine/models/__init__.py` (import `linear_eval` to register)
- Test: `tests/test_linear_eval.py`, extend `tests/test_models.py` for the evaluator change

**Interfaces:**
- Consumes: `Representation`, `target`, `score_to_cp` (Task 1); `iter_batches` (Task 2).
- Produces:
  - `EvaluationModel.expected_score(self, board: chess.Board) -> float` — expected score for **White** in `[0, 1]`; default implementation `cp_to_expected_score(self.predict_cp(board))` (unchanged behaviour of the existing models).
  - `ModelEvaluator.__call__` uses `model.expected_score(board)` and flips for Black, then clamps to `[0.01, 0.99]` (same results as today for existing models — test with the stub model).
  - In `linear_features.py`: `phase(board) -> float` in `[0, 1]` (1 = opening: `min(24, minors + 2*rooks + 4*queens) / 24`); `class LinearFeatures` (abstract): `name`, `size: int`, `def active(self, board: chess.Board, perspective: str) -> tuple[list[int], list[float]]` (sparse indices and values, already multiplied by phase for the middlegame block and by `1 - phase` for the endgame block); `class PstFeatures(LinearFeatures)`: per piece, index `block + piece_type_index * 64 + oriented_square`, value `+1` for the "own" side and `-1` for the other side with the square mirrored by rank, where "own" is the side to move (`perspective="side_to_move"`) or White (`"white"`, plus one turn feature `+1/-1`); blocks: middlegame `[0, 384)`, endgame `[384, 768)`; `size = 768` (+1 for the white perspective turn feature).
  - In `linear_eval.py`: `class LinearEvalModel(EvaluationModel)` with class attributes `features: LinearFeatures`, `artifact_suffix = ".npz"`; instance fields `weights: np.ndarray (size,)`, `bias: float`, `rep: Representation`; `logit(board) -> float` (pure Python sum over `active` with weights held as a Python list); `expected_score(board)` -> sigmoid, converted to White POV; `predict_cp(board)` via `score_to_cp`; `predict_cp_batch(boards)`; `save/load` (npz with `weights`, `bias`, and `rep`/`features.name` as JSON); `count_parameters() == size + 1`; `fit_stream(batches: Iterable[pd.DataFrame], validation: pd.DataFrame, config: dict) -> dict` (Adam on binary cross-entropy with soft targets from `target(...)`, L2 `config["l2"]`, `lr`, `batch_size`, `max_epochs`, `patience` on validation loss; returns summary with epochs, losses, seconds). The base-class `fit`/`predict_target` raise `NotImplementedError` with a pointer to `fit_stream`.
  - Registered model `@register_model("pst_tapered") class PstTaperedModel(LinearEvalModel)` with `features = PstFeatures()`, `architecture = "Tapered piece-square tables (768 weights), logistic"`.

- [ ] **Step 1: Write failing tests**: `test_phase_is_one_at_start_and_zero_with_bare_kings`; `test_pst_features_are_antisymmetric` (a position and its `board.mirror()` give the same active features under `side_to_move`); `test_mirrored_position_gets_complementary_white_score` (random weights: `expected_score(b) == pytest.approx(1 - expected_score(b.mirror()))`); `test_fit_learns_material` (synthetic batches whose target is the material balance through the Lichess curve: after fitting, a position up a queen scores > 0.9 for its owner); `test_save_load_round_trip_keeps_scores_and_representation`; `test_evaluator_unchanged_for_existing_models` (stub model: same values as before the change); `test_logit_matches_dense_computation` (pure-Python path equals `weights @ dense_vector + bias`).
- [ ] **Step 2:** Run — FAIL.
- [ ] **Step 3:** Implement. Fast path: precompute per-board the active indices with `board.piece_map()`; no numpy in `logit`.
- [ ] **Step 4:** Run the new tests and the full suite — PASS; ruff clean.
- [ ] **Step 5:** Commit `feat: linear evaluation core and tapered piece-square-table model`.

### Task 4: Training CLI for linear models (effort levels)

**Files:**
- Create: `src/chess_engine/training/effort.py`
- Modify: `src/chess_engine/training/train.py`
- Test: `tests/test_effort.py`, extend `tests/test_training.py`

**Interfaces:**
- Consumes: `LinearEvalModel.fit_stream` (Task 3), `iter_batches`, `load_sample` (Task 2), `update_card`, `today`, `current_commit` (`training/cards.py`).
- Produces:
  - `def effort_config(effort: int, available: int) -> dict` — keys `positions` (`round(100_000 * (available / 100_000) ** (effort / 100))`, capped by `available`), `max_epochs` (`1 + effort // 10`), `patience` (`2`), `trials` (`0` if `effort < 40` else `1 + (effort - 40) // 15`), `batch_size` (`4096`); monotonic in `effort`; raises `ValueError` outside `0..100`.
  - `train.py`: new options `--effort INT`, `--representation JSON-or-name` (default the default `Representation`), `--raw DIR` and `--sidecar DIR` (defaults `data/lichess/raw`, `data/lichess/sidecar`). When the model class is a `LinearEvalModel`, it: computes `effort_config`; runs a timed calibration on 50,000 positions and prints the configuration plus an estimated total duration; runs `trials` random-search trials (`l2` log-uniform `[1e-7, 1e-3]`, `lr` log-uniform `[1e-3, 3e-2]`) on 10% of `positions`, keeps the best by validation loss, then the final fit; saves; writes a card with `representation`, `effort`, `effort_config`, `seconds`, `train_samples`. The existing CSV path for the old models is untouched.

- [ ] **Step 1: Write failing tests**: `test_effort_bounds` (`effort_config(0, 10**9)["positions"] == 100_000`, `effort_config(100, 10**9)["positions"] == 10**9`), `test_effort_is_monotonic`, `test_effort_rejects_out_of_range`, `test_train_cli_trains_pst_on_fixture_shards` (tmp raw + sidecar from Task 2 fixtures, `--effort 0`, artifact and card written with `effort == 0` and the representation).
- [ ] **Step 2:** Run — FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run — PASS; full suite; ruff.
- [ ] **Step 5:** Commit `feat: effort-based training for linear models`.

### Task 5: Judge (static accuracy and 1-ply best-move agreement)

**Files:**
- Create: `src/chess_engine/training/judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `EvaluationModel.expected_score`, `predict_cp_batch`; `white_score`, `LICHESS_K` (Task 1); `load_sample` (Task 2).
- Produces:
  - `def static_metrics(pred_white: np.ndarray, true_white: np.ndarray) -> dict` — `spearman`, `mae` (expected-score units), `n`.
  - `def one_ply_choice(model, board: chess.Board) -> chess.Move` — the legal move maximizing the mover's evaluation of the resulting position, computed with ONE `predict_cp_batch` call over the non-terminal children (works for every model, also the slow CNNs; any monotonic scale ranks the same); a move that mates ranks first, stalemate or other draw ranks as 0 cp.
  - `def judge(model, test: pd.DataFrame, agreement_n: int = 20_000) -> dict` — `static_metrics` on all of `test` (predictions from `predict_cp_batch` in chunks, converted with `white_score(pred_cp, nan, Representation())`) using Stockfish truth `white_score(cp, mate, Representation())` (always the Lichess winprob scale, whatever the model's own representation), plus `best_move_agreement` on the first `agreement_n` rows; every metric also reported for the `quiet` and `non_quiet` slices: `{"all": {...}, "quiet": {...}, "non_quiet": {...}}`.

- [ ] **Step 1: Write failing tests**: `test_spearman_perfect_and_reversed`; `test_one_ply_takes_mate_in_one` (fixture `MATE_IN_ONE` from `tests/conftest.py` -> `a1a8` with any model); `test_one_ply_prefers_winning_the_queen` (material stub model); `test_judge_reports_slices` (keys and `n` add up).
- [ ] **Step 2:** Run — FAIL.
- [ ] **Step 3:** Implement (scipy is not a dependency: Spearman via ranks with numpy).
- [ ] **Step 4:** Run — PASS; ruff.
- [ ] **Step 5:** Commit `feat: judge with rank correlation and best-move agreement by slice`.

### Task 6: Bake-off script

**Files:**
- Create: `scripts/bakeoff_representation.py`
- Test: extend `tests/test_scripts.py` (fixture run)

**Interfaces:**
- Consumes: Tasks 1-5; `run_round_robin` (Task 11) for the tie-break.
- Produces: `scripts/bakeoff_representation.py --raw --sidecar --train-positions 5000000 --test-positions 200000 --effort 30 --out docs/training/bakeoff.json` which trains `pst_tapered` for the variants below (one axis changed at a time from the default), judges each, writes JSON and `docs/training/BAKEOFF.md` (table: variant, spearman/MAE/agreement for all / quiet / non_quiet, training seconds), then plays a tie-break round-robin at fixed depth 3 between the two best variants by `all` best-move agreement (`--tiebreak-games 200`), and STOPS (no automatic choice; the report ends with "Decision: pending").
  Variants: `winprob k=0.00368 stm all` (default), `cp_linear stm all`, `winprob k=0.0025 stm all`, `winprob k=0.005 stm all`, `winprob k=0.00368 white all`, `winprob k=0.00368 stm quiet` (trained on quiet, judged on everything), `winprob k=0.00368 stm all, depth >= 20` (drops the 11% shallowest, noisiest Stockfish evaluations), `winprob k=0.00368 stm all, mates dropped` (rows with `mate` set removed: 17.7% of rows, saturated targets of far-away mates). For the last two, `load_sample` must also return the raw `depth` column and the script filters on `depth` / `mate` after loading.
- [ ] **Step 1:** Failing test `test_bakeoff_runs_on_fixture` (tiny shards, `--train-positions 2000 --test-positions 500 --tiebreak-games 2`, report written with 6 rows).
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS; ruff.
- [ ] **Step 5:** Commit `feat: representation bake-off script`.

## Track B — the other two models (after Task 3; parallel with Tasks 4-6)

### Task 7: Material model

**Files:** Modify `src/chess_engine/models/linear_features.py`, `src/chess_engine/models/linear_eval.py`; remove `src/chess_engine/models/linear_baseline.py`, its card/artifact/config and references (`models/__init__.py`, docs, `scripts/run_all.sh`, tests); Test `tests/test_linear_eval.py`.

**Interfaces:** Produces `class MaterialCountFeatures(LinearFeatures)` named `"material5"` (own-minus-other count of P, N, B, R, Q; one block, no phase split; `+1` tempo/turn as in PST) and `@register_model("material_linear") class MaterialLinearModel(LinearEvalModel)`.

- [ ] **Step 1:** Failing tests `test_material_features_count_differences`, `test_material_model_mirror_complement`, `test_linear_baseline_is_gone` (`"linear_baseline" not in available_models()`).
- [ ] **Step 2-4:** Run fail, implement, run pass; full suite (update tests that used `linear_baseline` to use `material_linear` trained on fixtures, or the stub); ruff.
- [ ] **Step 5:** Commit `feat: material linear model; remove linear_baseline`.

### Task 8: PST + classical terms model

**Files:** Modify `src/chess_engine/models/linear_features.py`, `src/chess_engine/models/linear_eval.py`; Test `tests/test_linear_eval.py`.

**Interfaces:** Produces `class PstTermsFeatures(LinearFeatures)` = PST blocks + one middlegame and one endgame weight per term (own minus other, oriented by perspective): passed pawns, doubled pawns, isolated pawns, bishop pair, rook on open file, rook on half-open file, king shield pawns (own pawns on the 3 files around the king, 1-2 ranks ahead), mobility per piece type N/B/R/Q (popcount of `board.attacks_mask(sq) & ~own_occupancy`). Pawn-structure terms are cached by `(board.pawns, board.occupied_co[WHITE], board.occupied_co[BLACK]) & pawns` in an LRU dict (max 100,000 entries). `@register_model("pst_terms") class PstTermsModel(LinearEvalModel)`.

- [ ] **Step 1:** Failing tests: one per term on a hand-made FEN (`test_passed_pawn_term`, `test_doubled_and_isolated`, `test_bishop_pair`, `test_rook_open_and_half_open_file`, `test_king_shield`, `test_mobility_counts_attacks_not_legal_moves`), `test_pst_terms_mirror_complement`, `test_pawn_cache_hit_returns_same_features`.
- [ ] **Step 2-4:** Run fail, implement, run pass; ruff.
- [ ] **Step 5:** Commit `feat: piece-square tables plus classical terms model`.

## Track C — comparison tooling (parallel with Tracks A and B; uses only existing interfaces plus `expected_score`)

### Task 9: Completed depth statistic

**Files:** Modify `src/chess_engine/search/game_algorithms.py` (`SearchStatistics` gains `completed_depth: int = 0` as the last field; `heuristic_alphabeta_iterative_deepening_search` sets it to the deepest iteration that was not truncated); Test `tests/test_advanced_search.py`.

- [ ] **Step 1:** Failing test `test_completed_depth_excludes_truncated_iteration` (tree fixture where the cut-off fires during depth 3 -> `completed_depth == 2`, `max_depth_reached == 3`), and `test_completed_depth_equals_max_depth_without_cutoff`.
- [ ] **Step 2-4:** Fail, implement, pass; `scripts/benchmark_search.py --compare docs/benchmarks/search_baseline.json` still identical; ruff.
- [ ] **Step 5:** Commit `feat: record the deepest completed iteration`.

### Task 10: Depth-accuracy profiler

**Files:** Create `src/chess_engine/training/tradeoff.py`, `scripts/profile_tradeoff.py`; Test `tests/test_tradeoff.py`.

**Interfaces:**
- Consumes: `IterativeModelBookBot` and `IterativeHeuristicBot`, `TimeManager`, `OpeningBook`, `SearchStatistics.completed_depth` (Task 9), `load_sample` (Task 2).
- Produces:
  - `def fixed_time_manager(seconds: float) -> TimeManager` — budget exactly `seconds`: `TimeManager(move_fraction=1.0, min_turn_time=seconds, max_turn_time=seconds, max_clock_share=1.0, extension=1.0, safety_margin=0.0)`; the bot is called with `time_limit=seconds`.
  - `def make_player(spec: str, time_manager: TimeManager, max_depth: int = 30)` — `spec` is a registered model name (-> `IterativeModelBookBot(model=spec, opening_book=OpeningBook({}), max_depth=max_depth, time_manager=time_manager)`) or `"h2"` / `"h3"` (-> `IterativeHeuristicBot` with that heuristic and the same time manager). The profiler passes `fixed_time_manager(seconds)`.
  - `def profile(spec: str, positions: pd.DataFrame, seconds: float) -> dict` — mean and median `completed_depth`, mean nodes/s, and `search_agreement` (share of positions where the chosen move equals `best_move`), overall and for quiet / non_quiet.
  - `scripts/profile_tradeoff.py --players pst_tapered,pst_terms,material_linear,h2,h3,mlp_features,... --seconds 1,3,10 --positions 300 --out docs/training/tradeoff.json` writes JSON and `docs/training/TRADEOFF.md` with the table and a plot `docs/img/depth_vs_accuracy.png` (x = completed depth, y = static best-move agreement from the cards or search agreement, one point per model and time; matplotlib, already in the `ml` extra). Accepts `--parallel N` (one player/time per process).
- [ ] **Step 1:** Failing tests `test_fixed_time_manager_budget_is_exact`, `test_make_player_builds_model_and_heuristic_players` (stub model registered in the test), `test_profile_reports_depth_and_agreement` (2 positions, 0.2 s, stub model).
- [ ] **Step 2-4:** Fail, implement, pass; ruff.
- [ ] **Step 5:** Commit `feat: depth-accuracy profiler at fixed time per move`.

### Task 11: Parallel round-robin

**Files:** Create `src/chess_engine/training/tournament.py`, `scripts/run_tournament.py`; remove `src/chess_engine/training/strength.py` and `--strength` from `training/evaluate.py` (superseded) with their tests; Test `tests/test_tournament.py`.

**Interfaces:**
- Consumes: `make_player` (Task 10), `GameManager(BotRegistry()).play_game(white, black, time_limit, time_increment, starting_position=fen)` -> `GameResult` with `winner in {"white", "black", None}`.
- Produces:
  - `def paired_openings(n_pairs: int, plies: int = 6, seed: int = 0) -> list[str]` — distinct FENs after `plies` random legal plies, none terminal.
  - `def run_round_robin(specs: list[str], openings: list[str], base: float, increment: float, workers: int, out: Path) -> dict` — every unordered pair plays every opening twice with colours swapped; games run in a `ProcessPoolExecutor(workers)`, each worker builds its own players (`make_player(spec, TimeManager())`: the default clock-aware budget, since games use a real clock); results appended to `out` as JSON lines as they finish (resumable: games already in `out` are skipped); a crashed game is recorded with `"error"` and excluded from scores.
  - `def summarize(results: list[dict]) -> dict` — per player score %, W-D-L, Elo relative to `h3` (`-400*log10(1/s - 1)` per pairing vs h3 and a simple iterative fit across all pairs), 95% intervals (normal approximation of the score).
  - `def size_for_budget(hours: float, workers: int, n_players: int, seconds_per_game: float) -> int` — openings per pair that fit the budget.
  - `scripts/run_tournament.py --players material_linear,pst_tapered,pst_terms,h2,h3 --base 10 --increment 0.1 --workers 5 --hours 10 --out data/tournament/games.jsonl` sizes the openings with a short calibration, runs, and writes `docs/training/TOURNAMENT.md` (cross table, scores, Elo, intervals).
- [ ] **Step 1:** Failing tests `test_paired_openings_are_distinct_and_playable`, `test_every_pair_plays_both_colours`, `test_resume_skips_finished_games` (pre-filled JSONL), `test_crashed_game_is_recorded_not_fatal` (player spec that raises), `test_summarize_scores_and_elo` (hand-made results), `test_size_for_budget`. Use tiny `base` (e.g. 2 s) and 2 workers with `h2`/`h3` players.
- [ ] **Step 2-4:** Fail, implement, pass; full suite; ruff.
- [ ] **Step 5:** Commit `feat: parallel round-robin tournament`.

### Task 12: Cards, history table, docs

**Files:** Modify `src/chess_engine/training/report.py`, `src/chess_engine/training/evaluate.py`, `docs/training/TRAINING_TASKS.md`, `scripts/run_all.sh`, `docs/ADDING_A_MODEL.md`, `README.md` (models section); Test `tests/test_training.py`.

**Interfaces:**
- Consumes: `judge` (Task 5), `profile` (Task 10), `summarize` (Task 11).
- Produces: `python -m chess_engine.training.evaluate --model NAME --raw ... --sidecar ...` writes card section `accuracy` (= `judge(...)` on the fixed test sample: first 200,000 test rows by `load_sample(..., seed=42)`) and `cost` (`us_per_eval`, median single-position `expected_score` time); `scripts/profile_tradeoff.py` and `scripts/run_tournament.py` write `tradeoff` and `tournament` sections into the cards of the models they cover; `report.py` renders one table: model, kind (`plays` / `accuracy record`), params, representation, effort, Spearman, MAE, best-move agreement (all / non-quiet), µs/eval, completed depth at 1/3/10 s, search agreement, round-robin score and Elo ± interval; models without a section show "—". `run_all.sh` becomes: prepare sidecars -> train the three linear models at `EFFORT` -> evaluate all models -> profile -> tournament -> report.
- [ ] **Step 1:** Failing tests `test_report_renders_new_columns` (cards fixture with all sections), `test_accuracy_record_models_marked` (old models).
- [ ] **Step 2-4:** Fail, implement, pass; ruff.
- [ ] **Step 5:** Commit `feat: model cards and history table for the trade-off comparison`.

## Track D — second phase: the best model against fewer-mates variants (after Tasks 4, 11 and the main tournament decision)

User request: after the main tournament, the best new model plays a **variant of itself trained with only 5% / 10% of the mate rows** of the data. *Assumption (confirm or correct):* "5/10% of the mates" = keep 5% (resp. 10%) of the rows whose evaluation is a forced mate, so mates drop from ~17.7% of the data to ~0.9% / ~1.8%; the alternative reading (mates making up 5-10% of the training data) is a one-number change of `mate_keep` and is not built.

### Task 13: Mate-undersampled variants (`mate_keep`) and tagged models

**Files:**
- Modify: `src/chess_engine/training/representation.py`, `src/chess_engine/training/sidecar.py`, `src/chess_engine/training/train.py`, `src/chess_engine/models/registry.py`, `src/chess_engine/models/linear_eval.py`, `src/chess_engine/training/cards.py`, `src/chess_engine/training/tradeoff.py`, `src/chess_engine/training/tournament.py`, `scripts/run_tournament.py`
- Test: `tests/test_representation.py`, `tests/test_sidecar.py`, `tests/test_training.py`, `tests/test_tournament.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 4, 10, 11.
- Produces:
  - `Representation.mate_keep: float = 1.0` (validated in `(0, 1]`; part of `name`, e.g. `...-mates5`, `to_dict`/`from_dict`).
  - `def keep_mate_rows(fens: pd.Series, mate_keep: float, seed: int = 42) -> np.ndarray` (bool mask: non-mate rows must be passed through by the caller; selection is a CRC32 hash of the FEN against `mate_keep`, so the 5% set is a subset of the 10% set).
  - `load_sample(..., mate_keep: float = 1.0)` and `iter_batches(..., mate_keep: float = 1.0)`: rows with `mate` set are kept only where `keep_mate_rows` is true; `n` counts rows after the filter. Default 1.0 = unchanged behaviour.
  - `train.py`: `--mate-keep FLOAT` (default 1.0; applied to the TRAIN stream and to the validation set used for early stopping, **never** to judge / profile / tournament samples, which always see all rows) and `--tag TAG`. With a tag, the artifact is `<name>@<tag><suffix>` and the card `<name>@<tag>.json`.
  - Model spec syntax `name@tag` everywhere a model is named by a string (`registry.split_spec(spec) -> (name, tag | None)`, `load_model("pst_tapered@mates5")`, `make_player`, tournament `check_specs`); a tagged model loads its tagged artifact and reports itself with the full spec string as its bot name.
  - `run_round_robin`: the pool-rebuild counter (`MAX_POOL_REBUILDS`) resets after every game that finishes, so a long unattended run only gives up after consecutive broken pools without progress (test it).
  - `scripts/run_tournament.py --anchor SPEC` (default `h3`): the Elo anchor and the player required in `--players`; `parse_args` no longer requires `h3` when another anchor is given.

- [ ] **Step 1: Write failing tests**: `test_mate_keep_is_nested` (5% set subset of 10% set subset of all; share kept within ±1% of the target on 200,000 synthetic FENs), `test_load_sample_mate_keep_filters_only_mates` (non-mate rows untouched, mate share drops), `test_iter_batches_mate_keep`, `test_train_cli_mate_keep_and_tag_write_tagged_artifact_and_card` (fixture shards with mates, `--tag mates5 --mate-keep 0.05`, artifact `pst_tapered@mates5.npz`, card records `mate_keep`), `test_judge_sample_ignores_mate_keep` (the evaluation path takes no `mate_keep`), `test_tagged_spec_round_trip` (`split_spec`, `load_model("pst_tapered@x")` loads the tagged file and not the untagged one), `test_tournament_accepts_tagged_players_and_custom_anchor`.
- [ ] **Step 2:** Run them — FAIL.
- [ ] **Step 3:** Implement as specified above; one line per touched module in the commit message body.
- [ ] **Step 4:** Run the new tests, the full suite, ruff — PASS.
- [ ] **Step 5:** Commit `feat: mate-undersampled model variants and tagged model specs`.

### Task 14: Execution runbook for a light executor (after Task 15, so paths and module names are final)

**Files:** Create `docs/training/RUNBOOK.md`.

**Interfaces:** Consumes the final CLIs of Tasks 4, 6, 10, 11, 12, 13 (read their `--help`, do not guess flags). Produces a runbook that a Haiku-level executor can follow with direct command execution and log reading only.

- [ ] **Step 1:** After Tasks 4, 6, 12 and 13 are merged, write the runbook: for every run (R1-R7 and the Track D phase) a **LAUNCH** step (exact command, detached with `nohup setsid`, log path, the healthy-start check and what an unhealthy start looks like) and a **COLLECT** step (which JSON / log lines to read, the exact numbers to copy, how to interpret them with fixed rules such as "an Elo difference is significant only if the 95% interval excludes 0", and the exact section of which `.md` file to fill: `DATA.md`, `BAKEOFF.md`, `MODELS_HISTORY.md`, `TRADEOFF.md`, `TOURNAMENT.md`, `RESULTS.md`). Decision points (bake-off choice, which model is "best", deleting dominated models) are marked **STOP: report to the supervisor**; the executor never decides them.
- [ ] **Step 2:** Dry-run every command of the runbook on small fixture data (`--help` for each and one tiny real call where cheap) and fix what does not run.
- [ ] **Step 3:** Commit `docs: execution runbook for training and tournaments`.

### Task 15: Repository restructure and final documentation pass (runs BEFORE Task 14, after Tasks 6 and 13 are reviewed)

**Files:** moves and edits across `src/chess_engine/{training,comparison,models}`, `scripts/`, `tests/`, `docs/`, `README.md`, `.gitignore`, `.github/workflows/ci.yml`, `pyproject.toml`.

**Interfaces:**
- Consumes: everything built so far.
- Produces (pure moves plus import / path / doc updates; no behaviour change):
  - New package `chess_engine.comparison` holding what compares models: `judge.py`, `tradeoff.py`, `tournament.py`, `evaluate.py`, `report.py`, `metrics.py`. `chess_engine.training` keeps what trains and prepares data: `configs/`, `dataset.py`, `effort.py`, `lichess.py`, `lichess_raw.py`, `representation.py`, `sidecar.py`, `stockfish.py`, `targets.py`, `train.py`. `cards.py` moves to `chess_engine.models.cards` (used by both). Dependency direction: `comparison` may import `training` and `models`; `training` and `models` never import `comparison`.
  - Module commands become `python -m chess_engine.training.train`, `python -m chess_engine.comparison.evaluate`, `python -m chess_engine.comparison.report`.
  - `scripts/` split into `scripts/data/` (`download_lichess_evals.py`, `fetch_lichess_raw.py`, `prepare_dataset.py`, `label_with_stockfish.py`), `scripts/bench/` (`benchmark_search.py`, `profile_tradeoff.py`, `run_tournament.py`, `bakeoff_representation.py`), `scripts/docs/` (`render_pattern_boards.py`); `run_all.sh` stays in `scripts/`. `scripts/plot_models.py` and its test are deleted (dead: reads the legacy benchmark section). The `sys.path.insert(... "src")` hacks in scripts are removed (the package is installed with `pip install -e`), every script keeps its `if __name__ == "__main__":` guard. All default paths computed from `__file__` still point at the same files (repo root moves by one level for scripts in subfolders).
  - Tests that take more than ~30 s (`tests/test_tournament.py` pool tests, the bake-off fixture test in `tests/test_scripts.py`) are marked `@pytest.mark.slow`; CI still runs everything.
  - Cleanups carried from reviews: `evaluate` uses `artifact_path(args.model)` for `size_kb`; wrap `tests/test_training.py` line 237; `docs/TACTICAL_PATTERNS.md` no longer names the removed `TacticalAlphaBetaBot` (say the detector is kept for a cheaper use); `docs/training/DATA.md` and the `training/__init__.py` docstring describe the raw-Parquet flow; `docs/ADDING_A_MODEL.md` says `--representation` takes a JSON object or `default`; `.gitignore`: replace the stale `models/artifacts/*.tmp` with `src/chess_engine/models/artifacts/*.tmp`, keep `*@bake-*`, and confirm with `git status --ignored` that `data/lichess/`, `.superpowers/` and egg-info are ignored while `data/opening_book.json`, `data/README.md`, `docs/benchmarks/*.json` and model artifacts/cards are tracked.
  - Final docs pass: README (layout section matches the tree, commands copy-pasteable), `docs/ARCHITECTURE.md` (package table and diagrams include `comparison`, `training` contents, `models.linear_*`), `docs/SEARCH_ALGORITHMS.md` (make/unmake, fast terminal detection, `completed_depth`), `docs/training/*.md`, `CHANGELOG.md`; every command quoted in the docs is run with `--help` and every internal link resolves (script check).

- [ ] **Step 1:** Run the full suite and `ruff` first (baseline green), then do the moves with `git mv`, fix imports with a scripted rewrite, run the suite and ruff after every group of moves (training/comparison split; scripts split; test markers), commit per group (`refactor: ...`).
- [ ] **Step 2:** Add `tests/test_docs.py`: every relative Markdown link in `README.md` and `docs/**/*.md` (outside `docs/superpowers/`) resolves to an existing file; every `python -m chess_engine.<module>` quoted in the docs imports; every `scripts/...` path quoted exists.
- [ ] **Step 3:** Docs pass and cleanups above; run everything; commit `docs: final documentation pass and repository cleanup`.

## Track E — richer model families and the night pipeline (design: `docs/training/SELECTION_PROTOCOL.md`)

### Task 16: Slotted features and the king-bucket linear model (`pst_kb`)

**Files:** Modify `src/chess_engine/models/linear_features.py`, `src/chess_engine/models/linear_eval.py`; Test `tests/test_linear_eval.py`.

**Interfaces:**
- Consumes: `LinearFeatures` (`size(perspective)`, `active(board, perspective) -> (indices, values)`), `LinearEvalModel`, `register_model`.
- Produces:
  - `class SlottedPstFeatures(LinearFeatures)` name `"pst_slotted"`: 0/1 one-hot piece-square features for neural inputs. Under `side_to_move`, slot `0..383` = the mover's pieces `type*64 + oriented_square`, slot `384..767` = the opponent's pieces `384 + type*64 + oriented_square` (each colour seen from its own home rank, i.e. mirror by rank for Black's pieces, like `PstFeatures`), value always `1.0`; `size == 768`; attribute `max_active = 32`. Under `white` perspective: White pieces in `0..383`, Black pieces in `384..767`, no mirroring, plus one turn entry at 768 (`size == 769`, `max_active = 33`).
  - `class KingBucketFeatures(LinearFeatures)` name `"king_bucket"` (HalfKP-like, both sides, one-hot): for each side `p` in (mover = slot 0, opponent = slot 1) seen from its own home rank: `k` = its king square oriented to its own home rank; `bucket = 2*min(file, 7-file) + (1 if rank >= 2 else 0)` (8 buckets: 4 file classes x home/advanced); every NON-king piece of the position gets the index `(((slot*8 + bucket)*10 + piece_class)*64 + square)` where `piece_class = (piece_type-1) + (0 if friend of p else 5)` for types P,N,B,R,Q, and `square` = the piece square oriented to `p`'s home rank and **mirrored in file when the king file >= 4** (so the king always lies on files a-d); plus two plain king-square features at `10240 + slot*64 + king_square_oriented_and_file_mirrored_as_above`. `size == 10368` (`side_to_move`); `max_active = 64` (at most 30 non-king pieces on both sides, twice, plus 2 kings, bounded to 64: assert in a test). Under the `white` perspective: unsupported, raise `ValueError` with a clear message.
  - `@register_model("pst_kb") class KingBucketModel(LinearEvalModel)` with `features = KingBucketFeatures()`, `architecture = "King-bucketed piece-square features (10,368 weights), logistic"`.
- [ ] **Step 1: Write failing tests**: `test_slotted_features_are_one_hot_and_mover_relative` (start position: 32 entries, all `1.0`, White mover and the colour-mirrored position give the same sorted index list), `test_slotted_white_perspective_has_turn_entry`, `test_king_bucket_mirror_invariance` (a position and its `board.mirror()` give the same sorted index list under `side_to_move`), `test_king_bucket_file_mirror_symmetry` (a position and its file-mirror `chess.Board.transform(chess.flip_vertical)` ONLY when no castling rights, no en passant: same index list), `test_king_bucket_counts_and_bounds` (start position: 2*30 + 2 = 62 entries, all indices `< 10368`, `max_active` never exceeded on 200 random positions), `test_king_bucket_rejects_white_perspective`, `test_pst_kb_mirror_complement` (random weights: `expected_score(b) == approx(1 - expected_score(b.mirror()))`), `test_pst_kb_fit_learns_material` (small synthetic fit like the PST test: queen-up scores high for its owner), `test_pst_kb_save_load_round_trip`, and one test that `pst_kb` is registered.
- [ ] **Step 2:** Run them — FAIL. **Step 3:** Implement. **Step 4:** Tests + full suite + ruff — PASS. **Step 5:** Commit `feat: slotted and king-bucket features, king-bucket linear model`.

### Task 17: Neural evaluation models `nn4` and `nn8`

**Files:** Create `src/chess_engine/models/neural_eval.py`; Modify `src/chess_engine/models/__init__.py`; Test `tests/test_neural_eval.py`.

**Interfaces:**
- Consumes: `LinearEvalModel` (its featurization: `LinearFeatures.active`, `_encode` / `_encoded_frames` / the `workers` process pool, `expected_score`, `predict_cp`, `predict_cp_batch`, `DEFAULT_FIT_CONFIG` keys), Task 16 features (`SlottedPstFeatures`, `KingBucketFeatures`, `max_active`), `Representation`, Keras 3 (`keras`, TensorFlow CPU, already installed in the `ml` extra).
- Produces:
  - `class NeuralEvalModel(LinearEvalModel)` (it **is** a sparse-input evaluation model; subclassing reuses featurization, the process pool, perspective handling and every comparison tool): class attributes `features: LinearFeatures`, `hidden: Tuple[int, ...]` (sizes of the hidden layers; the first one is the accumulator). Weights live in NumPy arrays `layers: List[Tuple[np.ndarray, np.ndarray]]` (weight matrix, bias) with shapes `(features.size(rep.perspective), hidden[0])`, `(hidden[0], hidden[1])`, ..., `(hidden[-1], 1)`. Overrides: `logit(board)` (NumPy only, no Keras/TensorFlow import at play time: accumulator = `sum(value_i * W0[index_i]) + b0` over the active features, then clipped ReLU `min(max(x, 0), 1)` on the accumulator and plain ReLU on the other hidden layers, final linear layer), `count_parameters`, `save` (npz: `W0,b0,W1,b1,...`, `meta` JSON with feature-set name, representation, hidden sizes) and `load` (clear `ValueError` when the feature set or layer shapes do not match), `fit_stream(batches, validation, config)` with the same contract and summary keys as `LinearEvalModel.fit_stream` (`epochs, best_epoch, train_losses, val_losses, train_rows, seconds, config`), trained with Keras on CPU: build a `keras.Model` whose first layer is an embedding-sum over padded index/value arrays (index 0 = padding, features shifted by +1), the dense layers, a linear output; loss = binary cross-entropy with soft targets from the logit (`from_logits=True`); Adam with `lr` from the config; `l2` as weight decay on the accumulator; early stopping on validation loss with `patience`; the best epoch's weights are exported to the NumPy `layers`. Honour `workers` (featurization pool) exactly like the linear model and keep TensorFlow's intra-op threads at 2 in the training process so the pool is not starved. Initialisation: He-normal for dense layers, `N(0, 0.05)` for the accumulator.
  - `@register_model("nn4") class Nn4Model(NeuralEvalModel)`: `features = SlottedPstFeatures()`, `hidden = (256, 32, 32)` → weight layers 768→256→32→32→1 = **4 layers**, ~206,000 parameters.
  - `@register_model("nn8") class Nn8Model(NeuralEvalModel)`: `features = KingBucketFeatures()`, `hidden = (384, 128, 64, 64, 32, 32, 16)` → 10,368→384→128→64→64→32→32→16→1 = **8 layers**, ~4,000,000 parameters.
  - Both work with `training.train --effort`, `comparison.evaluate`, the profiler and the tournament **without any change in those modules** (they only use the `LinearEvalModel` interface); if one of them needs a change, make the smallest possible one and report it.
- [ ] **Step 1: Write failing tests**: `test_nn4_and_nn8_have_the_documented_layer_shapes_and_parameter_counts`, `test_logit_matches_a_dense_numpy_reference` (random weights, random positions: the sparse `logit` equals the dense reference computed with the same weights), `test_mirror_complement_holds_for_random_weights` (both models), `test_fit_stream_learns_material_and_lowers_validation_loss` (synthetic batches with material targets as in `test_linear_eval.py`, `nn4`, few hundred rows, 3 epochs; validation loss of the best epoch is lower than that of epoch 1), `test_fit_stream_workers_two_matches_one_worker_in_loss_shape` (not bit-exact: both finish and the first-epoch train loss is within 5%), `test_save_load_round_trip_gives_identical_scores`, `test_load_rejects_mismatching_shapes`, `test_play_path_does_not_import_tensorflow` (a fresh subprocess loads a saved tiny model and scores a board with `sys.modules` free of `tensorflow` / `keras`), `test_logit_cost_is_reported` (not asserted, prints µs per call for the report).
- [ ] **Step 2-4:** Fail, implement, pass; full suite, ruff. **Step 5:** Commit `feat: sparse-input neural evaluation models nn4 and nn8`.

### Task 18: The night pipeline script and status log

**Files:** Modify `scripts/run_all.sh` (or replace by `scripts/run_night.py` if clearer; keep `bash scripts/run_all.sh` working as the entry point); Test: `tests/test_scripts.py` (dry-run mode).

**Interfaces:**
- Consumes: the CLIs of Tasks 4, 5, 10, 11, 12, 13, 16, 17 (read their `--help`).
- Produces: one entry point `bash scripts/run_all.sh` with stages `sidecars → train → evaluate → profile → tournament → report`, each stage skipped when its outputs already exist (resumable), each logging to `data/runs/<stamp>/<stage>.log` and appending `{"stage", "status", "seconds", "detail"}` lines to `data/runs/<stamp>/status.jsonl`; a `DRY_RUN=1` mode prints the commands without running them. Variables with the defaults decided for **night 1**: `MODELS="material_linear pst_tapered pst_terms pst_kb nn4 nn8"` trained in that order, `EFFORT=30` for the linear models and `EFFORT_NN=30` for the networks (the networks get the same number of positions), `PROFILE_SECONDS="1,3"` with `PROFILE_POSITIONS=100`, `PLAYERS` for the tournament = `pst_tapered,pst_terms,pst_kb,nn4,nn8,h3t` (no `material_linear`, no `h2`; `h3t` is the tuned reference of Task 19) with `--anchor h3t`, `HOURS=4`, `BASE=60 INCREMENT=1`; evaluate runs on all of `$MODELS` plus the four old accuracy-record models (with `AGREEMENT_N=20000`, and `AGREEMENT_N_OLD=2000` for the CNNs). A failing stage stops the pipeline with a non-zero exit and a clear line in `status.jsonl`; a failing MODEL training does not stop the others (it is recorded and skipped by the later stages).
- [ ] **Step 1:** Failing test `test_run_all_dry_run_lists_every_stage_in_order` (dry run prints the stages and the model list). **Step 2-4:** implement, pass, `bash -n`, ruff. **Step 5:** Commit `feat: resumable night pipeline with status log`.

### Task 19: `h3t`, the tuned classical reference (the only heuristic in the tournament)

User decision: the tournament keeps ONE hand-crafted reference, not both `h2` and `h3`; it should be the most adapted one given its simplicity. Choice: keep the six `h3` features and fit their weights to Stockfish.

**Files:** Modify `src/chess_engine/evaluation/heuristics.py`, `src/chess_engine/comparison/tradeoff.py` (heuristics table), `src/chess_engine/comparison/tournament.py` (default anchor), `scripts/bench/run_tournament.py` (default players, anchor), `scripts/run_all.sh` (players, anchor, a `fit_h3t` stage before the tournament); Create `scripts/data/fit_h3t.py`; Test `tests/test_heuristics_tuned.py` (+ the existing tournament/script tests that name `h3`).

**Interfaces:**
- Consumes: `get_position_features(board) -> Dict[str, float]` and `FEATURE_NAMES` (six features in `[0, 1]` from White's point of view, `0.5` = equal), `load_sample`, `Representation` / `white_score`, the sidecars.
- Produces:
  - `H3T_WEIGHTS_PATH = <package>/evaluation/h3t_weights.json` (versioned, small): `{"weights": {feature: float}, "n": int, "r2": float, "r2_h3": float, "date": str}`.
  - `def h3t(state: ChessGameState, player: chess.Color) -> float`: `white = 0.5 + sum(w[name] * (features[name] - 0.5))`, clipped to `[0.01, 0.99]`, returned as `white` for White and `1 - white` for Black, so the evaluation is exactly complementary for the two players (weights may have any sign and need not sum to 1). Weights are loaded once and cached; if the file is missing raise `FileNotFoundError` naming `scripts/data/fit_h3t.py`.
  - `scripts/data/fit_h3t.py --raw --sidecar --positions 300000 --jobs 4 --out <path>`: samples TRAIN-split positions **without mates** with `load_sample`, computes the six features in a spawn process pool, fits `target_white - 0.5 ~ sum(w_i * (f_i - 0.5))` by least squares (no intercept) where `target_white = white_score(cp, nan, Representation())` (Lichess curve), reports R² of the fit and of the plain `h3` weights on a held-out 20% (VALIDATION-style split of the same sample by FEN hash), writes the JSON. Prints both R² so the gain over `h3` is visible.
  - `HEURISTICS = {"h2": h2, "h3": h3, "h3t": h3t}` in `tradeoff.py`; `tournament.ANCHOR` default and the default of `--anchor` in `run_tournament.py` become `h3t`; the default `--players` becomes `pst_tapered,pst_terms,pst_kb,nn4,nn8,h3t`; `h2` stays available as a spec but is in no default. `scripts/run_all.sh`: stage `fit_h3t` (skipped when `h3t_weights.json` is newer than the sidecars) runs right before the tournament, `PLAYERS` and `--anchor` use `h3t`.
- [ ] **Step 1: Write failing tests**: `test_h3t_is_complementary_for_white_and_black` (random positions, `h3t(s, WHITE) + h3t(s, BLACK) == approx(1)` unless clipped), `test_h3t_with_h3_like_weights_matches_h3_ordering` (weights set to `H3_WEIGHTS`: same ordering of 50 random positions as `h3`), `test_fit_recovers_known_weights` (synthetic: targets built from known weights + noise → fit within 0.05), `test_fit_script_runs_on_fixture_shards` (tiny raw shards + sidecars from the existing fixtures, writes the JSON to tmp, `r2` finite), `test_missing_weights_file_raises_with_hint`, `test_tradeoff_accepts_h3t_spec`, `test_run_tournament_defaults_use_h3t_anchor`.
- [ ] **Step 2-4:** Fail, implement, pass, full suite, ruff. **Step 5:** Commit `feat: h3t, the tuned classical reference, and the h3t-anchored tournament defaults`.

## Runs (coordinator; on real data, after the full download and Tracks A-C)

- [ ] **R1:** `scripts/prepare_dataset.py` on the full raw data; record counts (split sizes, quiet share) in `docs/training/DATA.md`.
- [ ] **R2:** Bake-off (Task 6) on real data. Coordinator reads `BAKEOFF.md`; user confirms the representation; record "Decision" in `BAKEOFF.md`.
- [ ] **R3:** Train `material_linear`, `pst_tapered`, `pst_terms` with the chosen representation at the effort the user picks (training prints the estimate first).
- [ ] **R4:** `evaluate` all models (including the four existing ones as accuracy records; for the CNNs lower `agreement_n` if the run is too slow and note it in the card).
- [ ] **R5:** `profile_tradeoff.py` for all models at 1, 3, 10 s.
- [ ] **R6:** Overnight `run_tournament.py` (~10 h) with `pst_tapered, pst_terms, h2, h3` at 60 s + 1 s (the default `TimeManager` has a 1.5 s minimum per move, which would turn a 10 s clock into scrambles and make depth-at-equal-time meaningless). `material_linear` is trained and measured in accuracy / depth profiles but does not play the tournament.
- [ ] **R7:** Apply the decision rule (strength at equal time decides; existing models are kept as records); remove dominated new models; regenerate `MODELS_HISTORY.md`; commit.
- [ ] **R8 (Track D, after R6/R7):** train the best new model as `<best>@mates5` and `<best>@mates10` with `--mate-keep 0.05` / `0.10` and the same effort; judge and profile them with the full model; run `run_tournament.py --players <best>,<best>@mates5,<best>@mates10 --anchor <best>` (~3-4 h, same time control) and document the head-to-head in `TOURNAMENT.md` and the accuracy / depth effect in `RESULTS.md`.
