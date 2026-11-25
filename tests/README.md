# Tests

The folders mirror the source: a test sits where the code it checks sits.

| Folder | Checks | Source |
|---|---|---|
| `app/test_uci.py` | the UCI client: the limits it sends and the moves of the binary | `src/chess_engine/uci.py` |
| `lab/models/` | model definitions, cards, saved files and tagged specs, the shipped models through the core, feature extractors, target representation and the export to the binary | `src/chess_lab/models/` |
| `app/gui/`, `app/test_uci.py` | the window (the widgets are skipped without PySide6), the UCI client | `src/chess_engine/{gui,uci.py}` |
| `core/` | the Rust core against the dense Python reference and the training graph (`test_parity.py`) | `rust/` |
| `lab/datasets/` | raw Lichess shards, perspective, sidecars, split, opening book and openings | `src/chess_lab/datasets/` |
| `lab/training/` | fitting the networks, effort, cache, the training command | `src/chess_lab/training/` |
| `lab/measure/` | judge, model evaluation, models table and the round robin | `src/chess_lab/measure/` |
| `lab/test_pipeline.py`, `lab/test_screenshots.py` | the pipeline script and the window captures | `src/chess_lab/{pipeline.sh,screenshots.py}` |
| `test_docs.py` | links, commands and script paths quoted in the docs exist | `README.md`, `docs/RESULTS.md`, `docs/MEASUREMENT.md` |

Shared pieces: `conftest.py` (fixtures, a material stub model, `untrained_model` and `random_net`, the `isolated_cards` fixture, and a guard that fails the run if a test
overwrites a shipped model file), `paths.py` (`ROOT`, `POSITIONS`, `SYZYGY`), `data/positions.json` (the
positions of the feature-parity tests), `data/syzygy/` (five 3-piece tables) and
`lab/models/positions.py` (the positions and raw-shard rows of the model tests).

## Rules

- A test checks one behaviour (a function, a format, a refusal). It does not train, play a full game or
  judge a model: that is the comparison's job. Anything heavy is replaced by a fake; the clock of the game
  loop is a fake too, so no test sleeps.
- Command-line modules are imported like any other module; their tests call their functions on a handful of
  synthetic positions.
- A test that takes more than a few seconds is marked `slow` (`pytest -m "not slow"` skips it).
- The Rust tests are the unit tests of each module (`cargo test --release`). The
  search tests (`rust/src/search/tests/`) play with a material network and a fake clock, so their values and
  times depend neither on a trained model nor on the machine. Values are whole centipawns and a mate is
  `MATE − plies`.

## Pitfalls

- **After touching `rust/`, run `pip install -e ".[dev]"` again.** Otherwise `pytest` tests the old compiled core.
  `cargo test` does not build the bindings (the `python` feature is off).
- **The features exist twice.** Training encodes with `src/chess_lab/models/features.py`, the core plays
  with `rust/src/eval/features/`; if they differ, one thing is trained and another played, with no error.
  `core/test_parity.py::test_core_score_matches_a_dense_numpy_reference` compares the core's score with a
  Python reference built from `features.py` for every model family: keep it green.
- **The models table of `docs/RESULTS.md` is generated** (`python -m chess_lab.measure.report`); a hand edit is
  lost, and `test_docs.py` does not check its numbers.
- **The first test that builds the Keras graph pays ~8 s of TensorFlow import** (`core/test_parity.py`); the rest of the suite stays under 3 s per test.
- **Timing figures need an idle machine.** Do not run the suite or compile while measuring speed.
