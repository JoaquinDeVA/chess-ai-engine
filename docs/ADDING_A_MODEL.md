# Adding an evaluation model

> **What it is:** A guide to adding an evaluation model, training it and comparing it with the others.
>
> **Read it when:** You want to train and measure a new model.
>
> **Next:** [Results](RESULTS.md)

An evaluation model answers one question: *how good is this position for White, as Stockfish
would see it?* The search uses the answer as its evaluation function. Models are compared in the
expected-score scale (win probability), at equal time per move, by the tools below.

## The linear models

A linear model is a sparse feature set and a logistic output. Adding one takes two small classes,
one in `src/chess_engine/models/linear_features.py` and one in `src/chess_engine/models/linear_eval.py`:

1. a `LinearFeatures` subclass with a `name`, `size(perspective)` (number of weights) and
   `active(board, perspective)`, which returns the indices and values of the active features of a
   position (`PstFeatures` and `MaterialCountFeatures` are complete examples);
2. a `LinearEvalModel` subclass registered by name, which points at the feature set:

```python
@register_model("my_model")
class MyModel(LinearEvalModel):
    """One-line description."""

    features = MyFeatures()
    architecture = "One-line description for the card"
```

The registry (`models/registry.py`) is what makes `my_model` a valid `--model` for the training
command, the evaluation tools and the bots.

## Train and measure

Each command works on one model, and `scripts/run_full_pipeline.sh` runs the whole flow for the
shipped ones (read its header for the steps and variables). The raw Lichess shards must already
exist in `data/lichess/raw`: see [DATA_AND_TRAINING.md](DATA_AND_TRAINING.md).

```bash
# sidecars (split, quiet flag, side to move) for the raw shards
python scripts/data/build_sidecars.py --raw data/lichess/raw --out data/lichess/sidecar

# train at an effort level 0-100 (effort N = N million positions, at least 1 M; epochs = 1 + N // 10)
python -m chess_engine.training.train --model my_model --effort 40

# judge on the fixed test sample: accuracy (Spearman, MAE, best-move agreement) and cost
python -m chess_engine.comparison.evaluate --model my_model

# depth reached and search agreement at a fixed time per move, against the reference bots
python scripts/bench/profile_depth_vs_accuracy.py --players my_model,h1 --seconds 1,3,10

# round-robin at equal time per move (default 10 s + 0.1 s); h1 is the Elo anchor and must be listed
python scripts/bench/play_tournament.py --players my_model,h1,linear_pst
```

`train` prints the chosen configuration and an estimated duration before it starts. Its options:

| Option | Meaning |
|---|---|
| `--effort` | training size and length, 0-100 (required for the linear models) |
| `--representation` | target scale: `default`, or a JSON object with the fields of `Representation` (see `training/representation.py`) |
| `--seed` | seed of the TRAIN sample |
| `--mate-keep` | share of mate positions kept, in (0, 1]; the default is 0.10 (`DEFAULT_MATE_KEEP` in `training/spec.py`) |
| `--tag` | trains a variant saved as `<model>@<tag>`, with its own card; `evaluate --model <model>@<tag>` judges it |

The default representation is called `winprob-k0.00368-stm-all` (win probability, side-to-move
point of view, all positions). A different share of mates adds a suffix to the name (`-mates5`,
`-mates100`); the default share adds none.

`evaluate` and `profile_depth_vs_accuracy.py` also accept the heuristic `h1` (`evaluate --model h1`),
which gives the classical evaluation a card to compare against.

## The model card

Every model has a card, `src/chess_engine/models/cards/<name>.json` (`<name>@<tag>.json` for a
variant), written by the commands above:

| Section | Written by |
|---|---|
| `training` | `training.train` |
| `accuracy`, `cost` | `comparison.evaluate` |
| `tradeoff` | `profile_depth_vs_accuracy.py` |
| `tournament` | `play_tournament.py`: one entry per time control, keyed `base+increment` (for example `60+1` or `600+5`), or `depthN` for a match limited by depth. A new tournament adds or replaces its own entry and keeps the others |

Retraining a model clears its old `training`, `accuracy`, `cost`, `tradeoff` and `tournament`
sections (`clear_measurements` in `models/cards.py`), because they describe the previous weights;
measure it again afterwards. `evaluate` and `comparison.report` regenerate the models table in
[RESULTS.md](RESULTS.md) from the cards; run `python -m chess_engine.comparison.report` to
refresh it by hand.

## Use it in a game

Once a model has a saved file, a bot can use it:

```bash
python main.py --player1 IterativeModelBookBot --model my_model --player2 IterativeHeuristicBot --depth 2
```

Without `--model` the bot uses `linear_pst_terms`, the best player in the tournaments (see
[Results](RESULTS.md)).

## The CSV-era models

`mlp_features` and `cnn_resnet` predate the linear flow. They are an `EvaluationModel` subclass
(`models/architectures.py`) that chooses an encoding from `models/dense_features.py` and builds a Keras
network (`models/keras_model.py`). They were trained on a CSV of Stockfish evaluations, and that
training code and data have been retired. **These two models are not in a fresh clone**: their
weights (`*.keras`) are ignored by git, so the bots and tools that load them fail until you have the
files in `src/chess_engine/models/artifacts/`. Their cards are kept in the repository as accuracy
records. `train` refuses them. With the weights in place they can still be measured from the raw
shards, but never retrained:

```bash
python -m chess_engine.comparison.evaluate --model mlp_features
```

## The contract

| Piece | Where | What it is |
|---|---|---|
| Feature set | `models/linear_features.py` (or `models/dense_features.py`) | turns a board into what the model consumes |
| Model class | `models/linear_eval.py` (or `models/*.py`) | `EvaluationModel` with `predict_cp`, `expected_score`, `fit`, `save`, `load` |
| Registration | `@register_model("name")` | makes it available to the CLIs and the bots by name |
| Card | `models/cards/<name>.json` | what is known about the model; written by the commands |
