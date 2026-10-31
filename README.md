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
- The search runs in Rust: in 10 seconds per move it reaches 7 plies where the Python version reached 4 or 5
  (8 to 35 times less time to a given depth). The tournament figures on this page were measured with the
  Python engine.
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
| Use the window | [The window](docs/GUI.md) |

The index of all pages is [docs/README.md](docs/README.md); the history of changes is the
[changelog](CHANGELOG.md).

## Quick start

Python 3.11 or newer and a Rust toolchain ([`rustup`](https://rustup.rs)): the search and the
evaluations are written in Rust and `pip` compiles them (a minute or two the first time).

```bash
git clone https://github.com/JoaquinDeVA/chess-ai-engine.git
cd chess-ai-engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # the engine and its tests
pip install -e ".[ml,data]"      # optional: Keras (trains the neural network) and the data tools
```

Two entries:

```bash
# the window: play against the engine, or watch two bots play (needs the gui extra)
pip install -e ".[gui]"
python main.py

# the UCI engine the window plays with (the window looks for it in rust/target/release/)
cargo build --release --manifest-path rust/Cargo.toml --bin uci
```

In the window you choose who plays White and Black (a person or an AI), the time control and, with the
**Settings** button of every AI, how it thinks (see [The window](docs/GUI.md)). Tournaments are
no longer in this repository: they are played with fastchess against the UCI binary, and the results
are in [Results](docs/RESULTS.md).

### Extras of the bots

The search bots come with everything that helps them play, and every extra can be switched off, for
example to compare evaluations alone.

| Extra | What it does | Switch | Default | Details |
|---|---|---|---|---|
| Opening book (`knowledge/opening_book.py`) | Plays the moves of `data/opening_book.json` (121,868 positions checked against Stockfish) while the game follows it; the Python side consults it before the engine | Settings → Opening book; `AISettings.opening_book` | on | [Data and training](docs/DATA_AND_TRAINING.md#the-opening-book-and-the-endgame-tables) |
| Endgame tables (`knowledge/endgame_tablebase.py`) | Plays positions of up to five pieces from the Syzygy tables in `data/knowledge/endgame`, without searching (used only if the folder exists); consulted before the engine | Settings → Endgame tablebases; `AISettings.tablebase` | on | [Algorithms](docs/algorithms/search-basics.md#endgame-tablebases) |
| Search techniques | Table move first, history, principal variation search, aspiration windows, null move and an incremental network (`lmr`, which lost, is off) | Settings → Advanced; UCI option `Techniques` | on (all but `lmr`) | [Algorithms](docs/algorithms/selective-techniques.md) |
| Several threads | The search runs on N threads that share one table (Lazy SMP); the search stops being deterministic | Settings → Advanced (threads); UCI option `Threads` | up to 4 | [Algorithms](docs/algorithms/rust-and-threads.md#several-threads-lazy-smp) |

## Reproduce the experiments

The data (about 24 GB) and the training are described in
[Data and training](docs/DATA_AND_TRAINING.md). Once the shards are in `data/lichess/raw`, the
flow, from the sidecars to the training of the models, the measurements and the models table, is one
script. Tournaments are not part of it; they are run with fastchess (see [Results](docs/RESULTS.md)):

```bash
DRY_RUN=1 bash src/chess_lab/scripts/run_full_pipeline.sh      # print every command without running anything
nohup setsid bash src/chess_lab/scripts/run_full_pipeline.sh > data/pipeline.log 2>&1 &
```

## Repository map

```text
rust/           the core: rules, search, evaluations, opening book and endgame tables (Rust)
src/chess_engine/
  uci.py        the UCI client the window plays through (book and tables first, then the binary)
  models/       h1, the learned evaluators and their cards
  knowledge/    the opening book and the endgame tables (Python faces of the Rust core)
  gui/          the window: board, game page, player selection and AI settings
src/chess_lab/  the laboratory (not needed to play)
  training/     sampling, targets and the training command
  measure/      judge, model evaluation and the results table
  scripts/      data pipeline and run_full_pipeline.sh
docs/           the documentation
tests/          the test suite
```

## Tests

```bash
pytest -m "not slow"
ruff check . && ruff format --check .
cargo test --manifest-path rust/Cargo.toml         # the Rust tests
cargo clippy --manifest-path rust/Cargo.toml --all-targets --features python -- -D warnings
```

## Credits

Joaquín De Vicente Abad. The base framework (game loop, bot interface)
and the first test suite come from the course *Inteligencia Artificial e Ingeniería del
Conocimiento* (Universidad CEU San Pablo). The search, the UCI engine, the models and the tools are
original work. The project uses [python-chess](https://python-chess.readthedocs.io),
[shakmaty](https://github.com/niklasf/shakmaty) and
[shakmaty-syzygy](https://github.com/niklasf/shakmaty-syzygy) (Rust), [PyO3](https://pyo3.rs),
[Stockfish](https://stockfishchess.org) and the
[Lichess evaluation database](https://database.lichess.org) (CC0).

## License

[MIT](LICENSE)
