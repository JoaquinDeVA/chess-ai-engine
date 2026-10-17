# Roadmap

> **Qué es:** Ideas pendientes, ordenadas por lo que más aportaría.
>
> **Cuándo leerla:** Antes de proponer o empezar una mejora.
>
> **Siguiente paso:** [Architecture](ARCHITECTURE.md)

Ideas, roughly in the order they would pay off.

## Evaluation models (redesign in progress)
Done: the linear models predict the expected score through a logistic target; positions come from
the raw Lichess evaluation database (sharded, with a fixed split); effort-based training; the
comparison measures accuracy, cost, depth at a fixed time per move and a round-robin at equal time.
Open:
- Run the comparison with the full data and decide which models stay (the rule is strength at
  equal time).
- Second round: the best model against variants trained with 5% or 10% of the forced-mate rows.
- Incremental feature updates (NNUE-style) instead of re-encoding the board at every node, only if
  the measured cost still dominates the search.
- Other model families under the same `EvaluationModel` contract, judged by the same comparison.

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
