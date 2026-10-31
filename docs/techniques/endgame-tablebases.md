# Endgame tablebases

> **What it is:** Playing positions of five pieces or fewer from the Syzygy tables instead of searching them.
>
> **Read it when:** You want to know how endgames are played or where the tables come from.
>
> **Next:** [The search in Rust](rust-core.md)

## Problem

A search with a static evaluation plays simple endgames badly. King and queen against king is a win that takes
about ten moves of method; at four or five plies the mate is beyond the horizon, the evaluation wanders and the
50-move rule can arrive first. Yet with five pieces or fewer the exact answer for every position is known.

## Idea

Look the position up instead of searching it. Syzygy tables give, for every position of up to five pieces, whether
the side to move wins, draws or loses, and the distance to the next capture or pawn move (DTZ). Pick the move with
the best result and, among those, the quickest progress.

## How it works

With five pieces or fewer (kings included) and no castling rights:

1. Keep the moves with the best result for the side to move (win, then draw, then loss).
2. If one of them mates at once, play it.
3. Winning: play the move with the smallest DTZ. Losing: the one with the largest, to resist longest.

**Worked example.** King and rook against king, White to move, three winning moves:

| move | result | DTZ after the move |
|---|---|---|
| `Rh7` | win | 15 |
| `Kd6` | win | 13 |
| `Ra1` | win | 19 |
| `Rh1` | draw (the rook can be taken) | - |

`Rh1` is dropped (draw), and `Kd6`, the winning move with the smallest DTZ, is played. No search runs.

## Implementation

`EndgameTablebase` in `rust/src/knowledge/tablebase.rs` (`covers`, `best_move`, the ranking), on top of the
`shakmaty-syzygy` crate, with `src/chess_engine/knowledge/endgame_tablebase.py` as its Python face. Consulted by
`UciBot.get_move` in `src/chess_engine/uci.py` after the [opening book](opening-book.md) and before the UCI binary,
only at the root: the search never probes the tables inside the tree. Switch: the window's Settings → Endgame
tablebases (`AISettings.tablebase`); used only if the tables are present.

The tables (3-4-5 pieces, 290 files, 939 MB) live in `data/knowledge/endgame`. They are not versioned and this
project cannot rebuild them: they come from `tablebase.lichess.ovh` or any Syzygy 3-4-5 set.

## Measured effect

Measured with the earlier Python engine (`scripts/bench/compare_endgame_tablebase.py --positions 60 --seconds 0.5`,
removed in `6162db8`; in the git history; report `data/reports/ENDGAME_TABLEBASE.md`): 60 random won positions of
three to five pieces, the defender plays the tablebase move, the winner has 0.5 s per move and 100 moves. With the
tables the winner mates in 52 of 60 (mean 14.2 moves); without them, in 8 of 60 (20.6 moves).

## Status

Adopted as an option of the window; since 2026-10-21 (`2fc0f86`). The UCI binary does not use the tables, so
engine matches compare search and evaluation only. The tables are permanent data, kept when the training data is
deleted.
