# Openings

> **What it is:** The suite of starting positions that engine matches are played from, and how it is built.
>
> **Read it when:** You run a match with fastchess or need to rebuild the suite.
>
> **Next:** [Results](RESULTS.md)

## Why a suite

Two deterministic engines that start from the initial position play the same game again and again, so a match of
a thousand games would measure one game. Each pair of games needs its own starting position, and that position
must not decide the result by itself: a suite of balanced positions a few moves into the game.

## How it is built

`src/chess_lab/scripts/data/build_openings.py` writes one FEN per line:

1. Draw random legal playouts of 8 to 10 half-moves from the start position, with a seeded generator (the same seed
   gives the same draw). Playouts that end the game or repeat a position already drawn are discarded.
2. Analyse every position with a local Stockfish at depth 12, one thread, and keep those within 0.5 pawns of
   equality.
3. Pick the requested number from the balanced ones, with the same seed.

The Lichess evaluation shards (`data/LichessPositions/raw`) are not the source: their FENs carry no move counters, so the
distance from the start cannot be read from them.

```bash
python src/chess_lab/scripts/data/build_openings.py --out data/openings/openings.epd   # seed 1, depth 12, 6000 playouts, 1000 kept
```

## How a match uses it

fastchess reads the file with `-openings file=data/openings/openings.epd format=epd order=random` and, with
`-games 2`, plays every opening twice with the colours swapped ([How a change is measured](MEASUREMENT.md)). The
`-srand` seed picks the order, so two matches with different seeds start from different openings.

## What was checked

The command above, run on 2026-10-31 with Stockfish 16, kept 1,431 of 6,000 playouts as balanced and wrote 1,000
openings (`sha256sum data/openings/openings.epd` = `27b6362df23485bbec341a06d7b385ee25775304e6c257c513205f13c3daecd9`).
The same arguments give the same file; `tests/lab/scripts/test_build_openings.py` checks the draw and the filter
with a fake engine. The three A/A matches of [Measurement](MEASUREMENT.md#the-aa-check) found no bias with this
suite; its effect against another suite is not measured.

The file is not versioned (`data/` is ignored); rebuild it with the command above.
