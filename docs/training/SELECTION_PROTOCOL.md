# Selection protocol and model families

How the candidate evaluation models are trained, compared and chosen over the two-night plan, and why.
Design background: the training and evaluation design (local planning record, not versioned).

## The risk in "train, pick the best, tune the best"

Picking one winner after a short, equal training and then spending the big budget on it has three
failure modes:

1. **Headroom bias.** A small linear model converges in minutes; a deeper network is still improving when
   a short run stops. At "80% of capacity" the small model looks better only because it is finished.
2. **Noisy verdict.** A tournament of a few hundred games per pair separates differences of about
   ±40 Elo. The ranking of close models is partly luck.
3. **Chained choices.** The data representation is chosen with one model family (piece-square tables) and
   then reused by families that may prefer another one.

## Protocol (two nights)

**Night 1 — broad, equal budget, with learning curves.**
- Every candidate trains on the *same* stream of positions with the *same* default representation
  (win probability, `k = 0.00368208`, side-to-move point of view, all positions) and the same number of
  positions. The validation loss is recorded after every pass over the stream (`val_losses` in the
  training summary stored in the card; effort 30 gives four points). The slope of the last passes is the
  **headroom** of each model: a curve still falling at the end means the model has not converged.
- Cheap screens first, games last: accuracy (`comparison.evaluate`), then depth and search agreement at
  fixed time per move (`scripts/bench/profile_tradeoff.py`), then a tournament among the contenders plus the
  anchor `h1`.
- Result: a ranking with confidence intervals, a headroom column, a cost column.

**Day / Night 2 — narrow, unequal budget.**
- Contenders = every model whose Elo interval overlaps the leader's **plus** the model with the most
  headroom (even if it is behind). Not just the winner.
- Representation bake-off (`scripts/bench/bakeoff_representation.py`) is run on the two most different
  contenders (the best linear model and the best network), so interactions between representation and
  capacity are visible. A representation is adopted only if it does not hurt either.
- Fine-tuning: higher effort, hyper-parameter random search on the validation split, for the contenders.
- Final: the fine-tuned contenders play each other **and their night-1 versions**, to prove the extra
  compute really bought strength. Mate-undersampled variants (`@mates5`, `@mates10`) of the best model
  play it last.
- Verdict rule: strength at equal time decides; a model stays only if it is best, or clearly better at
  something that matters (see the design).

## Is there a fundamentally better way?

We checked the alternatives; none beats this protocol for this machine and these data:

| Alternative | Why not (now) |
|---|---|
| Tune on game outcomes (Texel tuning with results) | The Lichess evaluation database has no game results; regressing on depth-28 Stockfish scores is the stronger signal anyway. |
| Optimise strength directly by self-play (SPSA / CLOP) | Tens of thousands of games per step; our budget is about 1,000 games per night. |
| Labels from our own search leaves (relabel with local Stockfish) | Better match to what the search sees, but ~hundreds of positions/s on 6 cores: millions of positions take days. Kept as a night-2 option **only if** the non-quiet slice of the judge shows a clear weakness. |
| Trees / gradient boosting | Better accuracy per feature, but no incremental update and ~10x slower per evaluation in Python: the opposite of the goal (depth). |
| Convolutional networks | 13 ms per evaluation: 100x slower than any candidate. They stay only as accuracy records. |

The two real improvements are protocol-level (learning curves and the headroom column; fine-tuning two or
more contenders; verifying night-2 models against night-1 versions) and model-level (the families below).

## Model families

All candidates are **sparse-input** models: a position activates at most ~32 (or ~64) features and the
score is a function of the sum of the active weight rows. That is why they are light, and why a move only
changes a handful of features (the incremental-update idea).

| Model | Input features | Hidden layers | Weight layers | Parameters | Role |
|---|---|---|---|---|---|
| `linear_material` (`material_linear`) | 5 piece-count differences | none | 1 | 6 | reference: what the squares add |
| `linear_pst` (`pst_tapered`) | 768 piece-square, phase-tapered | none | 1 | ~770 | main linear candidate |
| `linear_pst_terms` (`pst_terms`) | PST + 11 classical terms (pawns, mobility, king shield...) | none | 1 | ~790 | does knowledge per node pay for fewer nodes? |
| `linear_king_bucket` (`pst_kb`) | piece-square **relative to each king's bucket**, both sides (HalfKP-like) | none | 1 | ~10,000 | the *different* one: king-awareness without non-linearity |
| `net4_pst` (`nn4`) | the 768 `linear_pst` features | 256 (accumulator) → 32 → 32 | 4 | ~206,000 | the "4-layer" network |

`linear_king_bucket` versus `linear_pst` isolates the value of king-relative features; `net4_pst` versus `linear_pst`
isolates the value of non-linearity on the same cheap inputs.

**Dropped: an 8-layer network (`nn8`, king-bucket inputs, ~4 million parameters).** It was designed on
request, but it is too heavy for this CPU-only machine (about 20x the parameters of `net4_pst`, slower to train
and ~2x slower per evaluation, which costs search depth). It stays as a night-2 option **only if** `net4_pst`
beats `linear_pst` clearly and still has headroom; the design is: `linear_king_bucket` features → 384 → 128 → 64 → 64 →
32 → 32 → 16 → 1. The accumulator layer (first layer) is the incremental part: summing the rows of the active
features, which can be updated per move; the following layers are small dense layers recomputed per
evaluation.

Inference is NumPy only (Keras is used to train and its weights are exported), so playing does not need
TensorFlow. Expected cost per evaluation: `net4_pst` ~100 µs, against ~56 µs (`linear_pst`) and
~117 µs (`linear_pst_terms`); the profile measures the truth.

## Night-1 budget

Training is "good enough", not exhaustive: a share of the training rows per model (effort 30 is 30 %), two to four passes,
validated every pass (about 80% of the attainable accuracy for the linear models and less for `net4_pst`, which
is exactly why the headroom column exists). Order of cost: the linear models take minutes, the networks
tens of minutes; the profile takes about 40 minutes for everyone; the tournament takes the rest of the night.
