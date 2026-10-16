# Results — night 1

Filled in by the executor of [RUNBOOK.md](RUNBOOK.md). Numbers are copied from the cards, JSON files and
logs named in each step; nothing is estimated by hand. Date: (fill in).

## Data

- Raw shards / rows: (fill in)
- Split (train / validation / test): (fill in)
- Quiet share / White to move: (fill in)
- Source note: first ~124.9 M lines of the Lichess evaluation database; the file is not randomly ordered.

## Reference `h3t`

| | R² on held-out sample |
|---|---|
| plain `h3` weights | (fill in) |
| fitted `h3t` weights | (fill in) |

Fitted weights: (fill in)

## Training

| model | parameters | positions | seconds | estimate printed | epochs | best epoch | val loss per epoch | headroom |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

## Accuracy (fixed test sample)

| model | kind | Spearman all / quiet / non-quiet | MAE all | best-move agreement all / non-quiet | agreement sample | µs per eval |
|---|---|---|---|---|---|---|
| | | | | | | |

## Depth against accuracy

| model | completed depth 1 s | completed depth 3 s | search agreement 3 s | depth vs `pst_tapered` | Spearman vs `pst_tapered` |
|---|---|---|---|---|---|
| | | | | | |

## Tournament

Time control, workers, games per pair, crashed games: (fill in)

| rank | model | score % | Elo vs `h3t` ± interval | games |
|---|---|---|---|---|
| | | | | |

## Findings

(at most 8 bullets, each with a number)

## Open decisions for the supervisor

- Which models go to night 2?
