# Remote worker

> **What it is:** How engine matches run on the desktop machine (WSL2, 8 cores) from the laptop.
>
> **Read it when:** You launch a match with fastchess or the desktop changes address.
>
> **Next:** [Openings](OPENINGS.md)

**Problem.** A match of hundreds of games at 10 s + 0.1 s needs the cores of the desktop, and the laptop must stay free.

**What it does.** Two scripts in `src/chess_lab/scripts/worker/` drive the worker over SSH (`HOST` and `PORT` override the defaults `devicente@192.168.1.124` and `2222`).

```bash
src/chess_lab/scripts/worker/deploy.sh   # copies rust/target/release/uci, fastchess and openings.epd to ~/worker/
src/chess_lab/scripts/worker/match.sh check -each tc=10+0.1 proto=uci -rounds 1 -games 2 -concurrency 1 -srand 1
```

`match.sh NAME [fastchess options]` plays engine A against engine B (both the copied binary) from `openings.epd`, and brings `match.pgn` and `match.log` back to `data/matches/NAME/`. Options given after the name are passed to fastchess.

**Implementation.** The binary and fastchess are built on the laptop (glibc 2.39) and copied: the worker has glibc 2.43 and no Rust toolchain, so they run unchanged. The worker holds Stockfish and python-chess too, so `build_openings.py` can also run there.

**Measured effect.** The command above ran 2 games at 10 s + 0.1 s in 56 s on the worker, result in `data/matches/check/match.log` (1 win, 1 loss for the same binary). fastchess warns "No info line available to extract score" because the binary prints no score in its `info` lines; it affects only adjudication, which is not used.

**Status.** Implemented. Known traps: the WSL2 address changes after a restart and the Windows port forward must be repeated ([Pitfalls](PITFALLS.md)); measure speed with the desktop idle.
