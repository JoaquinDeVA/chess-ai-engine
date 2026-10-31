# Decisions

> **What it is:** The design decisions still in force, each with the reason and the evidence.
>
> **Read it when:** You are about to change something that looks arbitrary, or wonder why it is the way it is.
>
> **Next:** [Architecture](ARCHITECTURE.md)

| Decision | Why | Evidence |
|---|---|---|
| Evaluation models take sparse input (a position activates ~32–64 features; the score is a function of the sum of the active weight rows). | Depth per second matters more than accuracy per evaluation; a move changes few features, which allows incremental update. Convolutional networks took ~13 ms per evaluation, about 100 times a linear model. | [Models](MODELS.md) |
| The search core is Rust (`shakmaty`, `shakmaty-syzygy`, PyO3, `maturin`); training, tournaments and the command line stay in Python. | Python reached depth 4–5 in 10 s. C or C++ would be as fast per node and need CMake and ctypes. | [Rust and threads](algorithms/rust-and-threads.md) |
| Several threads use Lazy SMP (same root, shared lock-free table, the main thread owns the clock). One thread in tournaments. | Splitting the root concentrates the work in the first move. One thread keeps tournaments reproducible and the clock fair. | [Rust and threads](algorithms/rust-and-threads.md) |
| The clock alone decides the search depth: no default `max_depth`, `min_depth` 1, technical cap `MAX_DEPTH` = 30; a fixed depth is an explicit option for benchmarks and depth tournaments. | The depth 3 limit dates from when evaluation was very slow. With `geometric_growth` the completed depths are identical with a floor of 1 or 3 (60 random positions), and repeating iterations costs fewer nodes than searching the last depth directly (730,876 against 1,083,355 up to depth 7). | [Search basics](algorithms/search-basics.md) |
| Quiescence search (4 plies) is on by default. | At depth 3 it raises the agreement with Stockfish by +0.047 (`h1`) and +0.096 (`linear_pst_terms`) on 1,000 positions, at 2.5–3.6 times the nodes. | `src/chess_lab/scripts/bench/compare_quiescence.py` |
| The bots use `incremental`, `tt_move`, `history`, `pvs`, `aspiration` and `null_move`; `lmr` is implemented and off. | Without `lmr` the set scored 76.5 % (+205 Elo) against the reference over 100 games; with aggressive `lmr` 63.5 %; with mild `lmr` 49.0 % against the set without it. Fewer nodes is not enough. | [Selective techniques](algorithms/selective-techniques.md) |
| Opening book and endgame tables are on in every search bot; tournaments and measurements switch them off (`make_player`). One book (`data/opening_book.json`) and the Syzygy 3-4-5 tables are permanent; training data is disposable. | Tournaments compare evaluations, not knowledge. The book is validated by sampling against Stockfish, not by games. | [Results](RESULTS.md) (book check) |
| Training keeps 10 % of the forced-mate positions (`DEFAULT_MATE_KEEP`). | A 100-game match, 60 s + 1 s, same model: 71.0 % against 29.0 % for the model trained with all the mates (+156 Elo). One pair of models, one effort. | [Data and training](DATA_AND_TRAINING.md) |
| Training effort is measured in millions of training positions (`--effort N` = N M, epochs = 1 + N // 10). | A unit that can be reasoned about and compared. Small linear models saturate early. | [Data and training](DATA_AND_TRAINING.md) |
| The reference time control is 10 min + 5 s. | The old claim that the weakest model won at 60 + 1 was a confound of different mate proportions. | [Results](RESULTS.md) |
| One reproducible pipeline (`src/chess_lab/scripts/run_full_pipeline.sh`, resumable with `.done` files, `DRY_RUN=1`); a model card holds one tournament entry per time control; retraining clears the old measurements. | Scripts per night and one result per card hid which control a number came from. | [Data and training](DATA_AND_TRAINING.md) |
| The window is PySide6 with two entry points (`python main.py` and `python -m chess_lab.comparison.play_tournament`). | The console and its options duplicated the window. The search releases the GIL, so the AI thinks in a `QThread`. | [The window](GUI.md) |
| `chess_lab` (training, comparison, scripts) is separate from `chess_engine` (the app); the lab imports the app, never the reverse. | The app held what runs when playing and what produces and measures models. | [Architecture](ARCHITECTURE.md) |
