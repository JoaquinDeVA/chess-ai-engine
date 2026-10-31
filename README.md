# Chess AI Engine

A chess engine with an alpha-beta search written in Rust and played through UCI. It asks an evaluation
function how good each position is: a hand-written heuristic, a logistic model or a small neural network
trained to imitate Stockfish on 125 million positions of the Lichess evaluation database. A Python window
plays against it, and a Python laboratory holds the rest of the experiment: the data pipeline, the training
and the measurements, with the commands that reproduce them.

[![CI](https://github.com/JoaquinDeVA/chess-ai-engine/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-green)

## What it found

Six evaluation functions played each other at equal time, from a 6-parameter material count to a
206,000-parameter network. These tournaments were played with the earlier Python engine; every figure and
its command are in [Results](docs/RESULTS.md#the-short-answer).

- The best player is the **Piece-square-and-terms logistic model** (`linear_pst_terms`, 791 parameters):
  66 % of the points at 10 min + 5 s, the best point estimate in all three tournaments (the intervals
  overlap those of the next models). It is the model the engine plays with.
- The neural network judges positions best (Spearman 0.700 against Stockfish) but does not win more games.
- Counting material alone is clearly the worst (21 % at 10 min + 5 s).
- The move of the search to Rust made it 8 to 35 times faster to a given depth
  ([the search in Rust](docs/techniques/rust-core.md)); six pruning and ordering techniques on top of it scored 76.5 %
  against the plain search ([search techniques](docs/techniques/README.md)).

## Where to read

| You want to | Read |
|---|---|
| Play against the engine | [The window](docs/GUI.md) |
| Understand how it chooses a move | [How the engine chooses a move](docs/SEARCH.md), then [Search techniques](docs/techniques/README.md) |
| Know what each evaluation model contains | [Models](docs/MODELS.md) |
| Change the code | [Architecture](docs/ARCHITECTURE.md) |
| Rebuild the data or retrain | [Training data](docs/DATA.md), [Training](docs/TRAINING.md) |
| Read or repeat a measurement | [Results](docs/RESULTS.md), [How a change is measured](docs/MEASUREMENT.md) |

Every page is listed in [docs/README.md](docs/README.md); the history of changes is the [changelog](CHANGELOG.md).

## Quick start

Python 3.11 or newer and a Rust toolchain ([`rustup`](https://rustup.rs)); `pip` compiles the Rust core
(a minute or two the first time).

```bash
git clone https://github.com/JoaquinDeVA/chess-ai-engine.git
cd chess-ai-engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,gui]"
cargo build --release --manifest-path rust/Cargo.toml --bin uci
```

Check the engine answers UCI:

```bash
printf 'uci\nisready\nposition startpos\ngo movetime 1000\n' | rust/target/release/uci
```

It lists its options and plays a move; the depth, nodes and move depend on the machine:

```text
id name chess_core
option name Hash type spin default 64 min 1 max 65536
option name Threads type spin default 1 min 1 max 64
option name Model type string default linear_pst_terms
option name Techniques type string default incremental,tt_move,history,pvs,aspiration,null_move
uciok
readyok
info depth 7 nodes 287757 time 788
bestmove g1f3
```

Then `python main.py` opens the window, where you choose who plays each colour, the time control and,
with the **Settings** button of every AI, how it thinks ([The window](docs/GUI.md)). The window plays the
[opening book](docs/techniques/opening-book.md) and the [endgame tablebases](docs/techniques/endgame-tablebases.md)
itself and asks the UCI binary for every other move. Any UCI interface (fastchess, Cute Chess) can play the
binary too. `pip install -e ".[ml,data]"` adds Keras and the data tools for training.

## Status

- Working: the Rust search with six techniques on, the UCI binary, the window, the training and measurement
  pipeline (`DRY_RUN=1 bash src/chess_lab/scripts/run_full_pipeline.sh` prints every command without running it).
- Engine matches are played with fastchess on a separate machine ([Measurement](docs/MEASUREMENT.md)); the model
  tournaments in [Results](docs/RESULTS.md) predate the Rust engine and are kept as history.
- Known gaps: the UCI binary ignores the increment (`winc`/`binc`); late move reductions (`lmr`) are
  implemented and off ([lmr](docs/techniques/lmr.md)).

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
