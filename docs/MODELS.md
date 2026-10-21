# Models

> **What it is:** A tour of every evaluation model in the engine: what kind of model it is, what it takes as input, what its parameters mean and what it learns.
>
> **Read it when:** You want to understand what is inside a model before reading its numbers, or before adding a new one.
>
> **Next:** [Results](RESULTS.md)

An evaluation model answers one question: given a chess position, how likely is the player whose turn it is to win? The search engine (see [ARCHITECTURE.md](ARCHITECTURE.md) and [ALGORITHMS.md](ALGORITHMS.md)) calls this answer at the leaves of its tree and uses it to compare moves. The models differ only in how they produce that probability. Each one is trained to imitate Stockfish evaluations from the Lichess database (see [DATA_AND_TRAINING.md](DATA_AND_TRAINING.md)), so what a model "learns" is always the same thing: a cheap approximation of what a very strong engine thinks of a position.

## The idea behind the linear models

Four of the models are, underneath, the same machine: a logistic regression over sparse features. A position is turned into a short list of "active" features (for example "I have 2 more knights than my opponent", or "my rook stands on e1"), each with a value. The model keeps one weight per feature, adds up `weight * value` over the active ones, adds a single bias, and squeezes the total through a sigmoid to get a probability between 0 and 1. The slides of a university course on machine learning call exactly this "logistic regression", and the code in `src/chess_engine/models/linear_eval.py` implements it as `sigmoid(w . x + b)`.

Training adjusts the weights so that the predicted probability matches the win probability derived from Stockfish's score. A positive weight means "having this is good for the side to move"; a negative weight means the opposite. The bias is the value of an empty, perfectly balanced feature list: it captures the small edge that simply having the move is worth.

All features are written from the point of view of the side to move: the board is mirrored when Black moves, and the opponent's pieces count with a minus sign. So the model returns the win probability of whoever is about to play, and the engine converts it back to White's point of view when needed.

The one thing a linear model cannot do is combine features. Its verdict is a plain sum, so every feature contributes the same amount whatever else is on the board. A king with no pawn shield is worth the same penalty with queens on the board and with queens already traded, even though in reality it is far more dangerous in the first case. Only the neural network below, which has hidden layers, can learn that kind of interaction.

A note on names: every model has a descriptive name, used in this page, and a short identifier, written in parentheses, which is what you type in the code and in the command-line tools. Two identifiers can carry a suffix such as `linear_pst_terms@quiet`, which names a second copy of the same model trained on different data; the structure described here is the same.

## Material logistic model (`linear_material`)

The simplest model there is. It counts, for each piece type, how many more pieces the side to move has than the opponent, and learns how much each piece type is worth. It knows nothing about where the pieces stand.

| | |
|---|---|
| Type | Logistic regression |
| Input features | 5 numbers: pawns, knights, bishops, rooks and queens, each as (my count minus opponent's count). Kings are skipped |
| Parameters | 6, computed from a fresh instance: 5 weights plus the bias |
| Meaning of each weight | The value of one extra piece of that type, on the logit scale |
| Output | Sigmoid of the weighted sum plus bias: win probability of the side to move |

A worked example, using the weights from the version at the time of writing (they change whenever the model is retrained): pawn 0.38, knight 0.63, bishop 0.75, rook 1.03, queen 2.12, bias 0.14. With equal material the sum is just the bias, and the sigmoid of 0.14 is about 0.53: a slight edge for having the move. If the side to move is a rook up, the sum is 0.14 + 1.03 = 1.17, and the sigmoid of 1.17 is about 0.76. Notice that the learned values come out in the familiar order pawn < knight < bishop < rook < queen, without anyone telling the model the textbook values.

## Piece-square logistic model (`linear_pst`)

This model adds location. Instead of "a rook", the feature is "a rook on e1", and each of those has its own weight. That lets it discover, for instance, that knights prefer the centre and rooks like the seventh rank.

| | |
|---|---|
| Type | Logistic regression |
| Input features | One feature per piece, per square, per game phase: 6 piece types x 64 squares x 2 phases |
| Parameters | 769, computed from a fresh instance: 768 weights plus the bias |
| Meaning of each weight | How much a piece of that type standing on that square is worth in that phase |
| Output | Sigmoid of the sum plus bias: win probability of the side to move |

Each piece contributes `+1` for the side to move and `-1` for the opponent, and Black's pieces are seen from their own home rank so both sides share the same table. The game phase is a number from 1 (all pieces on the board) down to 0 (bare kings), computed from the remaining knights, bishops, rooks and queens. Every piece activates two features at once: one in the middlegame block, scaled by the phase, and one in the endgame block, scaled by one minus the phase. The model therefore holds two tables per piece (a middlegame one and an endgame one) and blends them smoothly. This is how a king can be told to hide early and to walk to the centre late.

## Piece-square-and-terms logistic model (`linear_pst_terms`)

It keeps everything from the piece-square model and adds eleven classical chess ideas that a table of squares cannot express by itself, such as pawn structure and piece activity.

| | |
|---|---|
| Type | Logistic regression |
| Input features | The 768 piece-square features plus 11 terms, each split into a middlegame and an endgame part (22 more) |
| Parameters | 791, computed from a fresh instance: 790 weights plus the bias |
| Meaning of each weight | 768 as in the piece-square model; the other 22 are the value of one unit of a term in the middlegame and in the endgame |
| Output | Sigmoid of the sum plus bias: win probability of the side to move |

The terms, in the order of `TERM_NAMES`, are:

- `passed`, `doubled` and `isolated`: the difference in passed, doubled and isolated pawns between the two sides.
- `bishop_pair`: owning both bishops.
- `rook_open` and `rook_half_open`: rooks on files with no pawns at all, or with no friendly pawn.
- `king_shield`: friendly pawns standing in front of the king.
- `mobility_knight`, `mobility_bishop`, `mobility_rook` and `mobility_queen`: how many squares those pieces attack that are not occupied by their own side.

Every term is the side to move's value minus the opponent's, and is blended by game phase exactly like the piece squares. What it learns beyond the previous model is how much each of these ideas is worth, and that the answer differs between middlegame and endgame (a passed pawn matters much more late).

## King-bucket logistic model (`linear_king_bucket`)

Here the idea is that a piece is worth different amounts depending on where the kings are. A knight near the enemy king is a different thing from the same knight far from it. The model gets this by keeping a separate piece-square table for each "king situation".

| | |
|---|---|
| Type | Logistic regression |
| Input features | One feature per piece class, square, king bucket and side: 10 x 64 x 8 x 2, plus 2 x 64 for the kings themselves |
| Parameters | 10,369, computed from a fresh instance: 10,368 weights plus the bias |
| Meaning of each weight | How much a piece of that class on that square is worth, given where the king of that side stands |
| Output | Sigmoid of the sum plus bias: win probability of the side to move |

A bucket is a coarse description of where a king stands. The board is mirrored so the king always sits on files a to d, and the king is placed in one of eight buckets: four classes of file distance from the edge, each either at home (the first two ranks) or advanced. Both kings choose a bucket, one for the side to move and one for the opponent, and every non-king piece is registered twice, once seen from each king. Pieces are split into friends and enemies of the side whose king defines the bucket (10 classes: 5 piece types times 2). This design is close in spirit to the king-relative inputs of modern chess engines. It is still a sum, so it learns more context-dependent piece values than the plain tables, but it does not learn interactions between arbitrary features. It is defined only for the side-to-move perspective.

## Four-layer piece-square network (`net4_pst`)

This is the only model that can combine features. It reads the same kind of piece-and-square information, but instead of summing weights straight into a probability it passes them through hidden layers that can build intermediate concepts.

| | |
|---|---|
| Type | Feed-forward neural network with a sparse first layer |
| Input features | 768 one-hot piece-square inputs: 6 piece types x 64 squares for the side to move, and the same for the opponent. No phase split |
| Layers | 768 inputs, an accumulator of 256, then 32, 32 and 1 |
| Parameters | 206,177, computed from a fresh instance (weights and biases of all four weight layers) |
| Meaning of the parameters | The first layer holds 256 numbers for each of the 768 piece-square inputs; the later layers mix those 256 numbers into 32, then 32, then one score |
| Output | A single logit, passed through the same sigmoid: win probability of the side to move |

A position activates at most 32 inputs (one per piece). The first layer, the accumulator, simply adds up the 256-number rows of the active inputs and clips the result to the range 0 to 1. The remaining hidden layers use ReLU activations, and the last layer is linear. Because the second and third layers receive all 256 accumulated values at once, a hidden unit can respond to "king exposed and queens present" and stay silent otherwise, which is precisely the combination the linear models are blind to. Evaluation at play time uses only NumPy, with no deep-learning framework, so it stays fast enough to run inside the search; Keras is used only to train the weights.

## The hand-written heuristic (`h1`)

`h1` is not a trained model with a model file, and it is not in the model registry. It is a classical evaluation function written by hand: six readable features, each normalised to the range 0 to 1 where 0.5 means "equal", combined with a weighted sum. Its code is in `src/chess_engine/evaluation/heuristics.py`.

| | |
|---|---|
| Type | Hand-designed linear combination (no sigmoid, no bias) |
| Input features | Six differences between White and Black: material, positional (hand-written piece-square tables), mobility, centre control, king safety and development |
| Parameters | 6 weights, stored in `src/chess_engine/evaluation/h1_weights.json` |
| Meaning of each weight | How strongly that feature pulls the evaluation away from 0.5 |
| Output | `0.5 + sum(weight * (feature - 0.5))`, clipped to 0.01 to 0.99 |

The features are designed by a person, but the six weights were fitted to Stockfish evaluations by a small script, so the heuristic also "learns", just very little: it can only choose how much to trust each of the six ideas. Weights may be negative, and a feature can end up with almost no weight at all if it does not help predict Stockfish. The clipping keeps a heuristic value strictly between 0.01 and 0.99, so it can never be mistaken for a real win or loss, which are scored exactly 1 and 0. It serves as the classical baseline that the learned models are compared with.

## The two course models

`mlp_features` and `cnn_resnet` come from a university course and are different in kind: they are Keras networks, and their weight files are not versioned in the repository, so a fresh clone does not have them. They are never retrained here, only measured, to show where the home-made models stand against them.

`mlp_features` is a small dense network, 29 inputs, then layers of 32 and 16 units with ReLU, then one output (1,505 parameters, counting weights and biases of that layout). Its 29 inputs are hand-crafted numbers: turn, game phase, material, piece counts, pawn advancement, centre occupation, king distance to the centre, check flags, castling rights and mobility. `cnn_resnet` is a residual convolutional network that reads the board as an 8x8 grid with 12 piece planes plus 9 extra numbers, and whose code documents 572,353 parameters. Both learn to predict the same Stockfish-derived target as the rest.

For accuracy, speed and tournament numbers of every model above, see [Results](RESULTS.md).
