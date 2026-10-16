# Changelog

## Unreleased

### Removed
- `RandomBot`, `HeuristicAlphaBetaBot`, `MLBot`, `TacticalAlphaBetaBot`, `NonTacticalAlphaBetaBot`
  and the `FixedDepthBot` base. `PrologPatternDetector` moved to `knowledge/pattern_detector.py`.

### Changed
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
