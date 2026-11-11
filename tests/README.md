# Tests

The folders mirror the source: a test sits where the code it checks sits.

| Folder | Checks | Source |
|---|---|---|
| `app/test_uci.py` | the UCI client: the limits it sends and the moves of the binary | `src/chess_engine/uci.py` |
| `lab/models/` | model definitions, saved files and tagged specs, the shipped models through the core, feature extractors, target representation and the export to the binary | `src/chess_lab/models/` |
| `app/gui/`, `app/test_uci.py` | the window (the widgets are skipped without PySide6), the UCI client | `src/chess_engine/{gui,uci.py}` |
| `core/` | the Rust core against the Python feature extractors of `chess_lab/models/` | `rust/` |
| `lab/datasets/` | raw Lichess shards, sidecars, split, opening book and openings | `src/chess_lab/datasets/` |
| `lab/training/` | fitting the networks, effort, cache, the training command | `src/chess_lab/training/` |
| `lab/measure/` | judge, model evaluation, report and the round robin | `src/chess_lab/measure/` |
| `lab/test_pipeline.py`, `lab/test_screenshots.py` | the pipeline script and the window captures | `src/chess_lab/{pipeline.sh,screenshots.py}` |
| `test_docs.py` | links, commands and script paths quoted in the docs exist | `README.md`, `docs/RESULTS.md`, `docs/MEASUREMENT.md` |

Shared pieces: `conftest.py` (fixtures, a material stub model, and a guard that fails the run if a test
overwrites a shipped model file), `paths.py` (`ROOT`, `POSITIONS`, `SYZYGY`), `data/positions.json` (the
positions of the feature-parity and technique tests), `data/syzygy/` (five 3-piece tables) and
`lab/models/positions.py` (the positions and raw-shard rows of the model tests).

## Rules

- A test checks one behaviour (a function, a format, a refusal). It does not train, play a full game or
  judge a model: that is the comparison's job. Anything heavy is replaced by a fake; the clock of the game
  loop is a fake too, so no test sleeps.
- Command-line modules are imported like any other module; their tests call their functions on a handful of
  synthetic positions.
- A test that takes more than a few seconds is marked `slow` (`pytest -m "not slow"` skips it).
- The Rust tests live in `rust/tests/` (`cargo test --release`).

## Pitfalls

- **After touching `rust/`, run `pip install -e ".[dev]"` again.** Otherwise `pytest` tests the old compiled core.
  `cargo test` does not build the bindings (the `python` feature is off).
- **The features exist twice.** Training encodes with `src/chess_lab/models/features.py`, the core plays
  with `rust/src/eval/sparse_features.rs`; if they differ, one thing is trained and another played, with no error.
  `core/test_feature_parity.py` compares them on 300 positions per feature family: keep it green.
- **The models table of `docs/RESULTS.md` is generated** (`python -m chess_lab.measure.report`); a hand edit is
  lost, and `test_docs.py` does not check its numbers.
- **Timing figures need an idle machine.** Do not run the suite or compile while measuring speed.
