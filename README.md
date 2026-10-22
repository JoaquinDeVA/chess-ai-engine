# Chess AI Engine

A chess engine in Python. It searches with alpha-beta and asks an evaluation function how good each
position is. The evaluation can be a hand-written heuristic, a logistic model or a small neural
network trained to imitate Stockfish on 125 million positions of the Lichess evaluation database.
The repository also holds the whole experiment: the data pipeline, the training, the tools that
compare the models and the results, with the commands that reproduce them.

[![CI](https://github.com/JoaquinDeVA/chess-ai-engine/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code style](https://img.shields.io/badge/code%20style-ruff-261230)

## What it found

Six evaluation functions play each other at equal time, from a 6-parameter material count to a
206,000-parameter network trained to imitate Stockfish on 125 million positions.

- The best player is a **logistic model on piece-square features and a few positional terms**
  (`linear_pst_terms`, 791 parameters): 66 % of the points at 10 min + 5 s, first at 60 s + 1 s and
  at a fixed depth as well. The neural network judges positions better (Spearman 0.700 against
  Stockfish) but does not win more games.
- Counting material alone is clearly the worst (21 % at 10 min + 5 s).
- Training for longer does not help the simplest model: effort 90 gives the same accuracy as 30.
- Quiescence search adds up to 10 points of agreement with Stockfish at the same depth, and the
  endgame tables convert 52 of 60 won endgames where the search alone converts 8.

All the numbers, with the command behind each, are in [Results](docs/RESULTS.md).

## What is inside

| You want to | Read |
|---|---|
| See how the engine is organised | [Architecture](docs/ARCHITECTURE.md) |
| Understand how it chooses a move, and what each technique saves | [Search algorithms](docs/ALGORITHMS.md) |
| Know where the training data comes from and how to retrain | [Data and training](docs/DATA_AND_TRAINING.md) |
| Know what each evaluation model contains | [Models](docs/MODELS.md) |
| Read the measurements | [Results](docs/RESULTS.md) |
| Add a model of your own | [Adding an evaluation model](docs/ADDING_A_MODEL.md) |

The index of all pages is [docs/README.md](docs/README.md); the history of changes is the
[changelog](CHANGELOG.md).

## Quick start

Python 3.11 or newer.

```bash
git clone https://github.com/JoaquinDeVA/chess-ai-engine.git
cd chess-ai-engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # the engine and its tests
pip install -e ".[ml,data]"      # optional: Keras and the data tools (training and the neural network)
```

Play against a bot, or watch two bots play (moves in UCI notation, for example `e2e4`):

```bash
# you play White against the engine with the hand-written heuristic
python main.py --player1 human --player2 IterativeHeuristicBot

# two bots; the model decides how the second one evaluates positions
python main.py --player1 IterativeHeuristicBot --player2 IterativeModelBookBot --model linear_pst_terms --depth 4

# every bot against every other, saving the results and the games in PGN
python main.py --mode tournament
```

`python main.py --list-bots` shows the bots. `--position` starts from a FEN and `--export` saves the
game in PGN.

## Reproduce the experiments

The data (about 24 GB) and the training are described in
[Data and training](docs/DATA_AND_TRAINING.md). Once the shards are in `data/lichess/raw`, the
whole flow, from the sidecars to the training of the models, the measurements, the tournaments and
the results table, is one script. It takes roughly a day on a six-core machine, most of it the 10-minute tournament (about 13 hours):

```bash
DRY_RUN=1 bash scripts/run_full_pipeline.sh      # print every command without running anything
nohup setsid bash scripts/run_full_pipeline.sh > data/pipeline.log 2>&1 &
```

## Repository map

```text
src/chess_engine/
  search/       the game abstractions and the alpha-beta search
  bots/         the playable bots
  evaluation/   hand-crafted features and the heuristic h1
  models/       the learned evaluators and their cards
  training/     sampling, targets and the training command
  comparison/   judge, trade-off profile, tournament and the results table
scripts/        data pipeline, experiments (data/, bench/) and run_full_pipeline.sh
docs/           the documentation
tests/          the test suite
```

## Tests

```bash
pytest -m "not slow"
ruff check . && ruff format --check .
```

## Credits

Joaquín De Vicente Abad. The base framework (game manager, bot interface, registry, abstract game
classes) and the first test suite come from the course *Inteligencia Artificial e Ingeniería del
Conocimiento* (Universidad CEU San Pablo); the abstract classes derive from the code of
[AIMA](https://github.com/aimacode/aima-python) (MIT). The search, the bots, the models and the
tools are original work. The project uses [python-chess](https://python-chess.readthedocs.io),
[Stockfish](https://stockfishchess.org) and the
[Lichess evaluation database](https://database.lichess.org) (CC0).

## License

[MIT](LICENSE)
