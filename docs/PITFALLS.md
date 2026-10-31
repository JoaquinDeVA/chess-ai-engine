# Pitfalls

> **What it is:** Things that already cost time, as symptom → cause → what to do.
>
> **Read it when:** You are about to launch a long script, a tournament or a training run, or touch `rust/`.
>
> **Next:** [Decisions](DECISIONS.md)

## Launching scripts and experiments

- **An option does not do what its name says** (`--epochs` ignored on the `--effort` path; `--max-depth` not reaching the tournament calibration; retraining with another `--mate-keep` and no `--tag` would overwrite the main model). Read `--help` and the code that consumes each option, then do a minimal dry run (effort 1, one opening, 2 s).
- **Defaults point to an old layout** (reports, sidecars, caches). Check that `--raw`, `--sidecar`, `--out`, `--report` and `--output` point to what is current and that the size of what will be read matches the documentation.
- **An experiment overwrites another's results.** Each tournament rewrites the `tournament` section of the model cards. Check which files a command writes (artifacts, cards, cache, report) and restore with `git checkout` what should not change.
- **A new tournament silently reuses an old one.** `play_tournament` resumes by (white, black, opening) and does not look at the model version. Check that `--out` does not exist or is the one to resume; rename old ones with a date.
- **Models trained with different configurations are compared.** Read `training.representation` and the effort of each card first.
- **A card keeps measurements of the previous model after retraining.** `clear_measurements` empties them on training; before comparing, check the date and commit of each section.
- **Two long processes write the same log.** Use absolute paths, launch once, check with `ps`, stop by PID.
- **Stopping a tournament or training with `kill` of the parent leaves `multiprocessing` workers orphaned** (alive two days later at 2.3 GB each). Afterwards run `ps -eo pid,ppid,etime,rss,args` and kill by PID those with `ppid 1`.
- **Measuring speed under load.** `src/chess_lab/scripts/bench/measure_search_speed.py` uses the clock: do not run the suite or compile meanwhile, and compare with the same `--workers`.

## Search

- **11 % of the moves of a 10 min + 5 s tournament were played at depth 2 with a 40 s budget.** The `ratio_squared` estimator (last²/previous) explodes when depth 1 takes microseconds on a warm table. The default is `geometric_growth`. When judging a search change, look at `completed_depth` in the per-move records (`data/tournament/*/…json`), not only the result.
- **The public Lichess `cloud-eval` returns HTTP 429** and blocks for about a minute. Do not work around it: use local Stockfish (`check_opening_book.py --engine /usr/games/stockfish`).

## Code

- **After touching `rust/`, `pip install -e ".[dev]"` again.** Otherwise `pytest` tests the old binary. `cargo test` does not build the bindings (feature `python` off).
- **Two implementations of the features.** Training encodes with `chess_lab/models/linear_features.py`, the core plays with `rust/src/eval/features.rs`; if they differ, one thing is trained and another played, with no error. `tests/core/test_feature_parity.py` compares them on 300 positions per feature family.
- **The models table of `docs/RESULTS.md` is generated** (`python -m chess_lab.measure.report`); editing it by hand is lost. After a rename or a command change, grep the whole repository before saying the docs are in sync.
