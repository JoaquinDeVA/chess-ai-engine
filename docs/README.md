# Documentation

> **What it is:** The index of every page, grouped by what you came to do.
>
> **Read it when:** You do not know where to start, or you look for one specific page.
>
> **Next:** [Architecture](ARCHITECTURE.md)

The project is a chess engine with an alpha-beta search written in Rust, played through UCI, that asks an
evaluation function how good a position is. The engine plays with a small neural network (NNUE) trained to imitate
Stockfish on 125 million positions; a hand-written heuristic stays as the anchor of the measurements. The window and the
laboratory that trains and measures the models are in Python.

## To play

| Page | What you learn |
|---|---|
| [The window](GUI.md) | The board, the game page, the players and the AI settings |
| [Results](RESULTS.md) | Which model plays best, at which time control, and how every number was obtained |

## To understand how it plays

| Page | What you learn |
|---|---|
| [How the engine chooses a move](SEARCH.md) | Minimax, alpha-beta, the values, the path of one move |
| [Search techniques](techniques/README.md) | Every technique added to the search, its status and what it gained, one page each |
| [Models](MODELS.md) | What each evaluation model is, what it contains and what it learns |

## To change or measure it

| Page | What you learn |
|---|---|
| [Architecture](ARCHITECTURE.md) | The packages, what depends on what, the path of one move through the code |
| [Training data](DATA.md) | The Lichess evaluation database, the shards, the sidecars, the targets |
| [Training](TRAINING.md) | The effort scale, what training does, how to reproduce everything |
| [Measurement](MEASUREMENT.md) | How a change is judged: fastchess matches, paired openings, SPRT, the A/A check |
| [Openings](OPENINGS.md) | The suite of balanced starting positions for engine matches |

The installation and the commands to play are in the [README of the repository](../README.md).
The history of changes is in the [changelog](../CHANGELOG.md).

## How the pages are kept

A change of behaviour updates its page in the same commit. The models table in `RESULTS.md` is
generated from the model cards (`python -m chess_lab.measure.report`) and is never edited
by hand. Every number in these pages is followed by the command or the file it comes from, and
`tests/test_docs.py` checks that the links, the commands and the script paths they quote exist.
