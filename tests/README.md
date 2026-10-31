# Tests

The folders mirror the source: a test sits where the code it checks sits.

| Folder | Checks | Source |
|---|---|---|
| `app/bots/`, `app/engine/`, `app/search/` | bots and their registry, the game loop, the time budget and the estimators | `src/chess_engine/{bots,engine,search}/` |
| `app/models/` | the heuristic `h1`, the learned models and their target representations | `src/chess_engine/models/` |
| `app/knowledge/`, `app/gui/` | opening book, endgame tables, the window (the widgets are skipped without PySide6) | `src/chess_engine/{knowledge,gui}/` |
| `core/` | the Rust core against the Python feature extractors | `rust/` |
| `lab/training/` | sampling, split, effort, cards, the raw Lichess export | `src/chess_lab/training/` |
| `lab/comparison/` | judge, trade-off profile, tournament, standings | `src/chess_lab/comparison/` |
| `lab/scripts/` | one file per script, named `test_<script>.py` | `src/chess_lab/scripts/` |
| `test_docs.py` | links, commands and script paths quoted in the docs exist | `README.md`, `docs/` |

Shared pieces: `conftest.py` (fixtures, a material stub model, and a guard that fails the run if a test
overwrites a shipped model file), `paths.py` (`ROOT`, `POSITIONS`, `SYZYGY`), `data/positions.json` (the
positions of the feature-parity and technique tests) and `data/syzygy/` (five 3-piece tables).

## Rules

- A test checks one behaviour (a function, a format, a refusal). It does not train, play a full game or
  judge a model: that is the comparison's job. Anything heavy is replaced by a fake; the clock of the game
  loop is a fake too, so no test sleeps.
- Scripts are not a package: their tests load them by path with `importlib` and call their functions on a
  handful of synthetic positions.
- A test that takes more than a few seconds is marked `slow` (`pytest -m "not slow"` skips it).
- The Rust tests live in `rust/tests/` (`cargo test --release`).
