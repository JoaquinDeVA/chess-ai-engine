# Openings

> **What it is:** The suite of starting positions that engine matches are played from, and how it is built.
>
> **Read it when:** You run a match with fastchess or need to rebuild the suite.
>
> **Next:** [Results](RESULTS.md)

**Problem.** Two deterministic engines that start from the initial position play the same game again and again, so a match of a thousand games would measure one game. Each pair of games must start from a different position, and the position must not decide the result by itself.

**What it does.** `src/chess_lab/scripts/build_openings.py` writes a file with one FEN per line. fastchess reads it with `-openings file=data/openings/openings.epd format=epd order=random`, and plays every opening twice with the colours swapped.

**Implementation.**

1. Draw random legal playouts of 8 to 10 half-moves from the start position (a seeded generator, so the same seed gives the same draw). Playouts that end the game early or repeat a position already drawn are discarded.
2. Analyse every position with a local Stockfish at a fixed depth (12, one thread) and keep those scored within 0.5 pawns of equality.
3. Pick the requested number of openings from the balanced ones with the same seed.

The Lichess evaluation shards (`data/lichess/raw`) are not the source: their FENs carry no move counters, so the distance from the start cannot be read from them.

```bash
python src/chess_lab/scripts/build_openings.py --out data/openings/openings.epd   # seed 1, depth 12, 6000 playouts, 1000 kept
```

**Measured effect.** The command above, run on 2026-10-31 with Stockfish 16, kept 1431 of 6000 playouts as balanced and wrote 1000 openings (`sha256sum data/openings/openings.epd` = `27b6362df23485bbec341a06d7b385ee25775304e6c257c513205f13c3daecd9`). The same arguments give the same file; `tests/lab/scripts/test_build_openings.py` checks the draw and the filter with a fake engine. The effect of the suite on a match is not measured.

**Status.** Implemented. The file is not versioned (`data/` is ignored); rebuild it with the command above.
