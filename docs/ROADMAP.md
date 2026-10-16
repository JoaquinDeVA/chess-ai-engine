# Roadmap

Ideas, roughly in the order they would pay off.

## Evaluation models (needs its own design pass)
- Revisit the centipawn → expected-score conversion (today a linear map clipped at ±1267 cp; a
  logistic win-probability curve is the usual alternative).
- A much larger, deduplicated position set (the Lichess evaluation database is available through
  `scripts/download_lichess_evals.py`) and a fixed benchmark split.
- A speed budget: the search calls the evaluator thousands of times per move, so the cost per
  position (features included) matters as much as the accuracy.
- Incremental feature updates (NNUE-style) instead of re-encoding the board at every node.
- Light, full and future model families under the same `EvaluationModel` contract.

## Search
- Quiescence search to avoid evaluating positions in the middle of a capture sequence.
- Late-move reductions, null-move pruning, aspiration windows.
- Move ordering from the transposition table (best move of the previous iteration first).
- Parallel search.

## Tactical patterns
- Find an affordable use for `PrologPatternDetector`: move ordering only near the root, cached per
  position, or batched queries; measure nodes and time against the same search without it.

## Endgames
- Endgame tablebases. Syzygy only gives win/draw/loss and distance to zeroing (DTZ); a bot that
  converts won endgames needs distance-to-mate (DTM) data.

## Engine
- A UCI front end so the bots can play in standard chess GUIs.
- A bigger opening book generated from strong games.
