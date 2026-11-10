# Models

> **What it is:** A tour of the evaluation models of the engine: the default network (`net_pst_w32`), the other networks, the retired logistic model and the hand-written anchor, with what each takes as input and what was measured.
>
> **Read it when:** You want to understand what is inside a model before reading its numbers, or before adding a new one.
>
> **Next:** [Results](RESULTS.md)

An evaluation model answers one question: given a chess position, how likely is the player whose turn it is to win? The search engine (see [ARCHITECTURE.md](ARCHITECTURE.md) and [SEARCH.md](SEARCH.md)) calls this answer at the leaves of its tree and uses it to compare moves. The models differ only in how they produce that probability. Each one is trained to imitate Stockfish evaluations from the Lichess database (see [DATA.md](DATA.md)), so what a model "learns" is always the same thing: a cheap approximation of what a very strong engine thinks of a position.

## The idea behind the network (NNUE)

The engine evaluates with a small feed-forward network of the kind chess engines call NNUE (*efficiently updatable
neural network*). A position is turned into a list of active inputs, one per piece: "a knight of the side to move
on f3", "a pawn of the opponent on d5". There are 768 possible inputs (6 piece types × 64 squares × 2 sides), and a
position has at most 32 active. The first layer keeps one row of weights per input; its output is the sum of the rows
of the active inputs plus a bias, clipped to 0–1 (a *clipped ReLU*). One or two small layers follow, and the last one
gives a single number, the logit, that a sigmoid turns into the win probability of the side to move.

The first layer is the expensive one, and it is also the one that can be updated cheaply: a move changes two to four
pieces, so instead of adding up every row again the engine subtracts the rows of the inputs that disappear and adds
the rows of the ones that appear. That running sum is the *accumulator* ([incremental accumulator](techniques/incremental.md)).

Every input is written from the point of view of the side to move: its pieces come first, the opponent's second,
each colour seen from its own home rank. So the network returns the win probability of whoever is about to play,
and the engine converts it back to White's point of view when needed.

Unlike a sum of weights, the hidden layer can combine inputs: a king with no pawn shield can cost more with queens
on the board than without them. That is what the network gains over the logistic model that came before it.

A note on names: every model has a descriptive name, used in this page, and a short identifier, written in parentheses, which is what you type in the code and in the command-line tools. An identifier can carry a suffix such as `net_pst_w32@e30`, which names a variant of the same model trained with other options (`--tag`); the structure described here is the same.

## Piece-square network, width 32 (`net_pst_w32`): the default

| | |
|---|---|
| Type | Feed-forward network with a sparse first layer |
| Inputs → layers | 768 piece-square inputs → 32 → 1 |
| Parameters | 24,641 (`parameters` in its card) |
| Output | Sigmoid of the logit: win probability of the side to move |

**Training.** 30 M quiet positions (effort 30), 40 epochs, validation loss 0.63620 (card
`src/chess_lab/models/store/cards/net_pst_w32@e30.json`; `python -m chess_lab.training.train --model net_pst_w32
--effort 30 --lr-decay 0.5 --patience 3 --epochs 40 --tag e30`, the options its card records in `config`). Quiet means the side to move is not in check and the best move is not a capture or a
promotion ([Data](DATA.md)); the search resolves the others with quiescence.

**Measured effect.** On 200,000 validation rows (`python -m chess_lab.measure.evaluate --model net_pst_w32@e30`):
Spearman 0.693, MAE 0.118, 2.37 µs per evaluation in the laboratory harness. In the engine, `uci bench 5` gives
1,020,805 nodes/s (`rust/target/release/uci bench 5`, six-core laptop, 2026-11-09, at `8700e6f`). In games, SPRT
[0, 5] against the Piece-square-and-terms logistic model (`linear_pst_terms@e30`) in the round robin of 2026-11-06 to 2026-11-09 (fastchess, 10 s + 0.1 s, `Hash` 16, the 10,000 openings of `data/openings/openings.epd`, the default techniques of then):
+165.1 ± 32.9 Elo at 504 games, **H1 accepted** (`data/results/roundrobin/model_net_vs_linear/llr.csv`).

**Status.** Default since `8700e6f` (2026-11-09): the UCI binary embeds `rust/models/net_pst_w32.json`, exported with
`python -m chess_lab.scripts.export_uci_model --model net_pst_w32@e30 --out rust/models`; `setoption name Model value
<file>.json` loads any other exported network.

## Wider piece-square networks (`net_pst_w64`, `net_pst_w64x16`)

The same inputs with a wider first layer, and optionally a second hidden layer of 16.

| Id | Layers | Parameters |
|---|---|---:|
| `net_pst_w64` | 768 → 64 → 1 | 49,281 |
| `net_pst_w64x16` | 768 → 64 → 16 → 1 | 50,273 |

**Implementation.** `NnW32Model`, `NnW64Model` and `NnW64x16Model` in `src/chess_lab/models/definitions.py`;
evaluation in `rust/src/eval/sparse_net.rs`, without a deep-learning framework (Keras only trains the weights).

**Measured effect.** Cost of a full evaluation on real positions with artificial weights: 768 → 32 → 1 about
550 ns, 768 → 64 → 1 about 765 ns, 768 → 64 → 16 → 1 about 919 ns; the width of the first layer sets the cost.
Quality and playing strength: not measured.

**Status.** Registered, not trained.

## King-bucket piece-square network (`net_pst_k4_w64`)

**Problem.** One table of piece squares for every king position cannot say that a pawn on g2 matters more when the
own king is on g1 than when it is on c1; the hybrid networks add king buckets, but together with the classical terms,
which cost a computation at every leaf.

**What it does.** The 768 piece-square inputs of `net_pst_w32`, repeated in 4 blocks, one per bucket of the mover's
king: files a–d or e–h, by ranks 1–2 or 3–8, each side seen from its own home rank. Only the block of the current
bucket is active, so a position still has one entry per piece; no classical terms.

| Id | Layers | Parameters |
|---|---|---:|
| `net_pst_k4_w64` | 4 × 768 → 64 → 1 | 196,737 |

**Implementation.** `NnK4W64Model` in `src/chess_lab/models/definitions.py`; the `pst_k4` feature set,
`Kings4PstFeatures` in `src/chess_lab/models/features.py` and `FeatureSet::Bucketed` in
`rust/src/eval/sparse_features.rs` (parity: `tests/core/test_feature_parity.py`). The accumulator follows the moves and
is rebuilt when the king changes bucket ([incremental](techniques/incremental.md#how-it-works)).

**Measured effect.** Not measured.

**Status.** Registered, not trained.

## Hybrid networks (`net_hybrid_w32`, `net_hybrid_w64`, `net_hybrid_k4_w64`, `net_hybrid_k8_w64`)

**Problem.** A network on the 768 piece-square inputs alone has to learn from the board what eleven classical
terms (pawn structure, mobility, king shield; listed in `TERM_NAMES`, `src/chess_lab/models/features.py`) already say.

**What it does.** The 768 piece-square inputs plus the 22 values of the eleven terms (each weighted by the game
phase and by its complement), 790 inputs, one hidden layer and one logit. The `k4` and `k8` variants repeat the 768
piece-square inputs once per king bucket of the side to move (the bucket follows the square of its own king):
4 buckets are 2 file groups by 2 rank groups, 8 buckets are 4 file groups by 2 rank groups.

| Id | Inputs → layers | Parameters |
|---|---|---:|
| `net_hybrid_w32` | 790 → 32 → 1 | 25,345 |
| `net_hybrid_w64` | 790 → 64 → 1 | 50,689 |
| `net_hybrid_k4_w64` | 4 × 768 + 22 → 64 → 1 | 198,145 |
| `net_hybrid_k8_w64` | 8 × 768 + 22 → 64 → 1 | 394,753 |

**Implementation.** `NnHybridW32Model`, `NnHybridW64Model`, `NnHybridK4W64Model` and `NnHybridK8W64Model` in
`src/chess_lab/models/definitions.py`; the `pst_hybrid`, `pst_hybrid_k4` and `pst_hybrid_k8` feature sets in
`src/chess_lab/models/features.py` and `rust/src/eval/sparse_features.rs`. Since the general accumulator
([incremental](techniques/incremental.md#how-it-works)) the piece part is updated move by move, with a rebuild when the
king changes bucket, and the 22 term inputs are computed at every leaf; until then the whole first layer was summed at
every leaf, and the SPRTs below are from that time.

**Measured effect.** Accuracy, all trained at effort 30 on quiet positions and measured on the same laptop on 200,000
validation rows (`python -m chess_lab.measure.evaluate --model <id>`; the table in [Results](RESULTS.md#model-table)
is generated from those runs), and speed in the engine (`uci bench 5 data/results/uci_models/<id>.json`, laptop,
2026-11-09, at `8700e6f`):

| Model | Spearman | MAE | µs per evaluation | `uci bench` nodes/s |
|---|---:|---:|---:|---:|
| `net_pst_w32` (reference) | 0.693 | 0.118 | 2.4 | 1,020,805 |
| `net_hybrid_w32` | 0.702 | 0.115 | 2.8 | 600,912 |
| `net_hybrid_w64` | 0.718 | 0.111 | 2.9 | 528,562 |
| `net_hybrid_k8_w64` | 0.719 | 0.111 | 3.3 | not measured |
| `net_hybrid_k4_w64` | 0.722 | 0.110 | 3.3 | 507,407 |

SPRT [0, 5] in the round robin of 2026-11-06 to 2026-11-09 (fastchess, 10 s + 0.1 s, `Hash` 16, the 10,000 openings of `data/openings/openings.epd`, the default techniques of then) (`data/results/roundrobin/<job>/llr.csv`):

| Job | Players (A vs B) | Result |
|---|---|---|
| `model_width_64_vs_32` | `net_hybrid_w64` vs `net_hybrid_w32` | +44.0 ± 15.9 Elo, 1,612 games, **H1** |
| `model_terms_in_net_nodes` | `net_hybrid_w64` vs `net_pst_w32`, 250,000 nodes per move | +89.2 ± 22.5 Elo, 784 games, **H1** |

At a fixed 250,000 nodes per move (speed removed) the hybrid beats `net_pst_w32`; at a time control it is not yet settled. The
earlier time-control SPRTs of the hybrids and king buckets (`model_terms_in_net`,
`model_buckets_k8_vs_k4`, `best_net_hybrid_*_vs_linear`) were removed: they played without the accumulator against networks
that had it. The same questions are asked again with the same binary on both sides: `acc_hybrid_w64_vs_pst_w32`,
`acc_hybrid_k4_vs_w64`, `acc_hybrid_w64_vs_pst_w64`, `pst_*` (`queue.json` in `src/chess_lab/scripts/worker/`); results pending.
Data: `data/results/roundrobin/<job>/llr.csv`.

**Status.** Trained, not the default. They have an accumulator since 2026-11-09; their SPRT with it is pending.

## Piece-square-and-terms logistic model (`linear_pst_terms`): retired

A logistic regression, `sigmoid(w · x + b)`, over the 768 piece-square inputs (once for the middlegame and once
for the endgame) and the 22 values of the eleven classical terms: 791 parameters. Sparse input was chosen because
depth per second mattered more than accuracy per evaluation; the convolutional networks tried before it took about
13 ms per evaluation. It was a very solid base for a long time: the engine played with it from its first timed
tournaments to 2026-11-09, and most of the search techniques were measured with it. What it could not do is combine
features: its verdict is a plain sum.

What it taught that the networks keep: training on quiet positions only played +284 ± 43 Elo better than training on
every position (362 games at 60 s + 1 s, `data/results/matches/quiet_vs_actual/match.log`). More features (passed
pawns by rank, king attack, king-relative piece squares) gave no gain (+29.0 ± 35.2 and 0.0 ± 35.3 Elo in 300-game
screenings, `data/results/matches/{passed,king}_vs_base/match.log`).

**Last measurements.** `uci bench 5` with the binary of `a1bdf60` (laptop, 2026-11-09): 931,652 nodes/s, against
948,666 for `net_pst_w32` on the same binary. In games it lost the SPRT against `net_pst_w32` above (−165.1 ± 32.9
Elo at 10 s + 0.1 s); the 10 min + 5 s SPRT is pending.

**Status.** Retired in `8700e6f` (2026-11-09), with the whole linear fitting code of the laboratory; the last
version is in `a1bdf60`. Its cards stay in `src/chess_lab/models/store/cards/` with status `retired` and `superseded`.

## The hand-written heuristic (`h1`)

A classical evaluation written by hand (`rust/src/eval/classic_heuristic.rs`), not a model file: six differences between White and
Black (material, hand-written piece-square tables, mobility, centre control, king safety and development), each
normalised to 0–1, combined as `0.5 + sum(weight * (feature - 0.5))` and clipped to 0.01–0.99 so it is never taken
for a decided game. The six weights were fitted to Stockfish (`src/chess_lab/models/store/h1_weights.json`). It is the
anchor of the timed tournaments in [Results](RESULTS.md).

For accuracy, speed and match numbers of every model above, see [Results](RESULTS.md).
