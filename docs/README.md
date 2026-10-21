# Documentation

> **What it is:** The index of every page, in the order a newcomer should read them.
>
> **Read it when:** You do not know where to start, or you look for one specific page.
>
> **Next:** [Architecture](ARCHITECTURE.md)

The project is a chess engine written in Python: an alpha-beta search that asks an evaluation
function how good a position is. The functions it can ask range from a hand-written heuristic
to logistic models and a small neural network trained to imitate Stockfish on 125 million
positions. The pages below tell, in this order, how the engine is put together, how it searches,
where the training data comes from, what each model contains and what was measured.

| Page | What you learn | Read it when |
|---|---|---|
| [Architecture](ARCHITECTURE.md) | The packages, what depends on what, the path of one move | You are about to change the code |
| [Search algorithms](ALGORITHMS.md) | The alpha-beta search and, one section each, every technique added on top of it, with what it saved | You want to know how the engine chooses a move |
| [Data and training](DATA_AND_TRAINING.md) | The Lichess evaluation database, the shards, the targets, the effort scale, how to retrain | You want to rebuild the data or train a model |
| [Models](MODELS.md) | What each evaluation model is, what it contains and what it learns | You want to understand a model before reading its numbers |
| [Results](RESULTS.md) | Which model plays best, at which time control, and how every number was obtained | You want to compare models or repeat a measurement |
| [Adding an evaluation model](ADDING_A_MODEL.md) | The few classes a new model needs, and the commands that train and measure it | You want to add a model of your own |

The installation and the commands to play are in the [README of the repository](../README.md).
The history of changes is in the [changelog](../CHANGELOG.md).

## How the pages are kept

A change of behaviour updates its page in the same commit. The models table in `RESULTS.md` is
generated from the model cards (`python -m chess_engine.comparison.report`) and is never edited
by hand. Every number in these pages is followed by the command or the file it comes from, and
`tests/test_docs.py` checks that the links, the commands and the script paths they quote exist.
