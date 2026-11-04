# Models

> **What it is:** A tour of every evaluation model that is current in the engine: what kind of model it is, what it takes as input, what its parameters mean and what it learns.
>
> **Read it when:** You want to understand what is inside a model before reading its numbers, or before adding a new one.
>
> **Next:** [Results](RESULTS.md)

An evaluation model answers one question: given a chess position, how likely is the player whose turn it is to win? The search engine (see [ARCHITECTURE.md](ARCHITECTURE.md) and [SEARCH.md](SEARCH.md)) calls this answer at the leaves of its tree and uses it to compare moves. The models differ only in how they produce that probability. Each one is trained to imitate Stockfish evaluations from the Lichess database (see [DATA.md](DATA.md)), so what a model "learns" is always the same thing: a cheap approximation of what a very strong engine thinks of a position.

## The idea behind the logistic model

The logistic model is a logistic regression over sparse features. A position is turned into a short list of "active" features (for example "I have 2 more knights than my opponent", or "my rook stands on e1"), each with a value. The model keeps one weight per feature, adds up `weight * value` over the active ones, adds a single bias, and squeezes the total through a sigmoid to get a probability between 0 and 1. The slides of a university course on machine learning call exactly this "logistic regression", and `rust/src/eval/logistic_model.rs` implements it as `sigmoid(w . x + b)`. Sparse input was chosen because depth per second matters more than
accuracy per evaluation: a move changes few features, so the sum can be updated incrementally, while the
convolutional networks tried first took about 13 ms per evaluation, some 100 times a linear model.

Training adjusts the weights so that the predicted probability matches the win probability derived from Stockfish's score. A positive weight means "having this is good for the side to move"; a negative weight means the opposite. The bias is the value of an empty, perfectly balanced feature list: it captures the small edge that simply having the move is worth.

All features are written from the point of view of the side to move: the board is mirrored when Black moves, and the opponent's pieces count with a minus sign. So the model returns the win probability of whoever is about to play, and the engine converts it back to White's point of view when needed.

The one thing a linear model cannot do is combine features. Its verdict is a plain sum, so every feature contributes the same amount whatever else is on the board. A king with no pawn shield is worth the same penalty with queens on the board and with queens already traded, even though in reality it is far more dangerous in the first case. Only a network with hidden layers, like the candidates below, can learn that kind of interaction.

A note on names: every model has a descriptive name, used in this page, and a short identifier, written in parentheses, which is what you type in the code and in the command-line tools. An identifier can carry a suffix such as `linear_pst_terms@mates100`, which names a variant of the same model trained with other options (`--tag`); the structure described here is the same.

## Piece-square-and-terms logistic model (`linear_pst_terms`)

It scores every piece on every square, once for the middlegame and once for the endgame, and adds eleven classical chess ideas that a table of squares cannot express by itself, such as pawn structure and piece activity.

| | |
|---|---|
| Type | Logistic regression |
| Input features | The 768 piece-square features (6 piece types x 64 squares x 2 phases) plus 11 terms, each split into a middlegame and an endgame part (22 more) |
| Parameters | 791, computed from a fresh instance: 790 weights plus the bias |
| Meaning of each weight | 768 for the piece squares; the other 22 are the value of one unit of a term in the middlegame and in the endgame |
| Output | Sigmoid of the sum plus bias: win probability of the side to move |

The terms, in the order of `TERM_NAMES`, are:

- `passed`, `doubled` and `isolated`: the difference in passed, doubled and isolated pawns between the two sides.
- `bishop_pair`: owning both bishops.
- `rook_open` and `rook_half_open`: rooks on files with no pawns at all, or with no friendly pawn.
- `king_shield`: friendly pawns standing in front of the king.
- `mobility_knight`, `mobility_bishop`, `mobility_rook` and `mobility_queen`: how many squares those pieces attack that are not occupied by their own side.

Every term is the side to move's value minus the opponent's, and is blended by game phase exactly like the piece squares. What it learns is how much each of these ideas is worth, and that the answer differs between middlegame and endgame (a passed pawn matters much more late).

**Trained on quiet positions only.** A sum of weights cannot represent an exchange in progress, so positions where the side to move is in check or the best move is a capture or a promotion (see [Data](DATA.md)) only distort what it learns; the search resolves those with quiescence. The shipped model is trained on the quiet ones (effort 10 or 30, `python -m chess_lab.training.train --model linear_pst_terms --effort 10 --representation '{"filter":"quiet"}'`). Trained on quiet positions it plays +284 ± 43 Elo better than trained on every position (362 games at 60 s + 1 s, `data/matches/quiet_vs_actual/match.log`, [Results](RESULTS.md)). Three variants did not beat it: a steeper target (`k` 0.0065), keeping 0.5 % of the mates instead of 10 %, and 30 M positions (all within the error of 280 games).

**Limit.** Adding more features to it (passed pawns by rank, king attack, king-relative piece squares) gave no gain in play (+29.0 ± 35.2 and 0.0 ± 35.3 Elo in 300-game screenings, `data/matches/{passed,king}_vs_base/match.log`), so the logistic regression is at its limit here. Training on 30 M positions (`linear_pst_terms@e30`) raised Spearman from 0.638 to 0.646 and scored +39.6 ± 33.8 Elo in a 300-game screening (`data/matches/e30_vs_base/match.log`). Status: shipped.

## Light network candidates (`net_pst_w32`, `net_pst_w64`, `net_pst_w64x16`)

**Problem.** The logistic model cannot combine features, and a network large enough to do it well is too slow for the search: a 206 k-parameter network costs 3.8 µs per evaluation against 0.53 µs for the logistic model, and a node of the engine costs about 1.3 µs in total, so the evaluation dominates the node.

**What it does.** A feed-forward network with a sparse first layer: 768 slotted piece-square inputs, an accumulator of 32 or 64 units, optionally a second hidden layer of 16, and one logit that goes through the same sigmoid. The accumulator adds up the rows of the active inputs and can be updated move by move ([incremental accumulator](techniques/incremental.md)).

| Id | Layers | Parameters |
|---|---|---:|
| `net_pst_w32` | 768 → 32 → 1 | 24,641 |
| `net_pst_w64` | 768 → 64 → 1 | 49,281 |
| `net_pst_w64x16` | 768 → 64 → 16 → 1 | 50,273 |

**Implementation.** `NnW32Model`, `NnW64Model` and `NnW64x16Model` in `src/chess_lab/models/definitions.py`; evaluation in `rust/src/eval/sparse_net.rs`, without a deep-learning framework (Keras only trains the weights).

**Measured effect.** Cost of a full evaluation on real positions with artificial weights: the logistic model 526 ns, 768 → 32 → 1 about 550 ns, 768 → 64 → 1 about 765 ns, 768 → 64 → 16 → 1 about 919 ns. The width of the first layer sets the cost. Quality and playing strength: not measured, the networks are not trained yet.

**Status.** Registered, not trained, not shipped. The parameter range is to be fixed after measuring the cost per evaluation; the engine binary embeds only the logistic model.

## The hand-written heuristic (`h1`)

A classical evaluation written by hand (`rust/src/eval/classic_heuristic.rs`), not a model file: six differences between White and
Black (material, hand-written piece-square tables, mobility, centre control, king safety and development), each
normalised to 0–1, combined as `0.5 + sum(weight * (feature - 0.5))` and clipped to 0.01–0.99 so it is never taken
for a decided game. The six weights were fitted to Stockfish (`src/chess_lab/models/h1_weights.json`). It is the
anchor of the timed tournaments in [Results](RESULTS.md).

For accuracy, speed and match numbers of every model above, see [Results](RESULTS.md).
