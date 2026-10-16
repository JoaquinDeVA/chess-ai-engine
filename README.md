# Chess AI Engine

A chess engine in Python built around adversarial search: minimax, alpha-beta pruning and an advanced
iterative-deepening search, with hand-crafted, machine-learned and Prolog-assisted evaluation, plus a
game manager, tournaments and an extension framework for new bots and evaluation models.

[![CI](https://github.com/JoaquinDeVA/chess-ai-engine/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code style](https://img.shields.io/badge/code%20style-ruff-261230)

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Search algorithms](#search-algorithms)
- [Evaluation models](#evaluation-models)
- [Extending the engine](#extending-the-engine)
- [Testing](#testing)
- [Project history](#project-history)
- [Roadmap](#roadmap)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

The project combines three ways of deciding which position is better, on top of the same search:

* **Search**: alpha-beta with a transposition table, killer moves, MVV-LVA ordering and a clock-aware
  iterative deepening.
* **Evaluation**: weighted hand-crafted features, or models trained to imitate Stockfish (scikit-learn,
  Keras).
* **Knowledge**: an opening book and a Prolog knowledge base that recognizes pins, forks and skewers to
  order moves.

Everything plugs together through small interfaces, so a new bot is one file and a new evaluation model is
one class.

## Features

| Bot | Search | Evaluation | Needs |
|---|---|---|---|
| `IterativeHeuristicBot` | iterative deepening, transposition table, killers | weighted features (`h1`–`h4`) | – |
| `IterativeModelBookBot` | opening book, then iterative deepening | learned model | `.[ml]` |
| `HumanPlayer` | – | you, at the console | – |

## Architecture

```mermaid
flowchart TB
    CLI["main.py"] --> GM["engine/<br/>GameManager"]
    GM --> REG["bots/<br/>registry + bots"]
    REG --> SEARCH["adversarial_search/<br/>algorithms, time management"]
    REG --> EVAL["evaluation/<br/>features, heuristics"]
    REG --> MODELS["models/<br/>learned evaluators"]
    REG --> KNOW["knowledge/<br/>opening book, Prolog"]
    TRAIN["training/"] --> MODELS
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for class diagrams and the life of a move.

## Installation

Python 3.11 or newer.

```bash
git clone https://github.com/JoaquinDeVA/chess-ai-engine.git
cd chess-ai-engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"              # core engine and tests
pip install -e ".[ml,prolog,data]"   # optional: learned models, Prolog patterns, dataset tools
```

Optional system packages: SWI-Prolog for the tactical-pattern detector (`apt install swi-prolog`) and Stockfish for
`scripts/label_with_stockfish.py` (`apt install stockfish`).

## Usage

```bash
python main.py --list-bots

# you (White) against a bot, moves typed in UCI notation such as e2e4
python main.py --player1 human --player2 IterativeHeuristicBot

# bot against bot, choosing the model of the ML bot and the search depth
python main.py --player1 IterativeModelBookBot --model cnn_resnet --player2 IterativeHeuristicBot --depth 3

# from a given position (full FEN) and exporting the game
python main.py --player1 IterativeHeuristicBot --player2 IterativeHeuristicBot \
  --position "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1" --export game.pgn

# round-robin tournament between all bots (results and PGN files are written)
python main.py --mode tournament
```

`--model` and `--depth` are only applied to the bots that accept them. Time controls are set in
`config/settings.py`.

## Search algorithms

| Algorithm | Idea |
|---|---|
| Minimax / alpha-beta | textbook algorithms, kept for reference and testing |
| Heuristic alpha-beta | stops at a depth and evaluates with a heuristic |
| Advanced alpha-beta | transposition table with bound flags, killer moves, move ordering, faster wins preferred |
| Iterative deepening | deepens while the time budget allows and keeps the last *completed* iteration |

Details, flowcharts and the conventions used by the cut-off test are in
[docs/SEARCH_ALGORITHMS.md](docs/SEARCH_ALGORITHMS.md); the Prolog pattern detector is described in
[docs/TACTICAL_PATTERNS.md](docs/TACTICAL_PATTERNS.md).

## Evaluation models

Models estimate the evaluation of a position (centipawns, White's point of view) the way Stockfish
would. Four are included (`ridge_features`, `mlp_features`, `cnn_light`, `cnn_resnet`), plus a minimal
`linear_baseline` that demonstrates how to add one. Their comparison is generated in
[MODELS_HISTORY.md](MODELS_HISTORY.md). This part of the project is still being designed: the interface
is stable, the internals are not ([docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md)).

```bash
python scripts/download_lichess_evals.py --rows 1000000        # build data/evaluations.csv
python -m training.train    --model mlp_features --data data/evaluations.csv
python -m training.evaluate --model mlp_features --data data/evaluations.csv --strength
```

`bash scripts/run_all.sh` does this for every model; [TRAINING_TASKS.md](TRAINING_TASKS.md) is the checklist.

## Extending the engine

* **A bot**: one file in `bots/` implementing `evaluate()` →
  [docs/ADDING_A_BOT.md](docs/ADDING_A_BOT.md).
* **An evaluation model**: a registered `EvaluationModel` →
  [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md).
* **Another game**: implement `GameState` and `GameProblem`; the algorithms do not know about chess.

## Testing

```bash
pytest                       # whole suite (tests needing SWI-Prolog or Stockfish skip if absent)
ruff check . && ruff format --check .
```

The suite covers the search algorithms (with hand-made game trees), every bot, the registry and CLI,
the model layer, the training pipeline and the dataset scripts.

## Project history

```mermaid
timeline
    title From coursework to engine
    Sep 2025 : Tic-tac-toe with an optimal bot
    Oct 2025 : Alpha-beta, heuristics, iterative deepening
    Nov 2025 : Prolog tactical patterns
    Dec 2025 : Models trained on Stockfish evaluations, opening book
    2026 : Refactor into a documented, tested engine
```

## Roadmap

See [ROADMAP.md](ROADMAP.md): quiescence search, a redesign of the evaluation models (conversion of
evaluations, larger datasets, speed), endgame tablebases and a UCI front end.

## Authors

* Joaquín De Vicente Abad

## Acknowledgements

* The base framework (game manager, bot interface, registry, abstract game classes) and the initial test
  suite come from the *Inteligencia Artificial e Ingeniería del Conocimiento* course (CEU San Pablo
  University); the abstract game classes derive from the [AIMA](https://github.com/aimacode/aima-python)
  code (MIT). The search algorithms, bots, models and tooling in this repository are original work.
* [python-chess](https://python-chess.readthedocs.io), [Stockfish](https://stockfishchess.org),
  [SWI-Prolog](https://www.swi-prolog.org) with [PySwip](https://github.com/yuce/pyswip), and the public
  [Lichess evaluation database](https://database.lichess.org) (CC0).
* Alpha-beta image by Jez9999 on Wikimedia Commons (CC BY-SA 3.0).

## License

[MIT](LICENSE)
