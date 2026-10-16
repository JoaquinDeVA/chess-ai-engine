#!/usr/bin/env bash
# Retrain and re-benchmark every model, then refresh docs/training/MODELS_HISTORY.md and the plots.
#
#   bash scripts/run_all.sh                 # everything except the (GPU-sized) CNN retraining
#   TRAIN_CNN=1 bash scripts/run_all.sh     # also retrain cnn_resnet
#   GAMES=0 bash scripts/run_all.sh         # skip the playing-strength matches
#
# Variables: DATA (default data/evaluations.csv), GAMES (games per opponent, default 10),
#            DEPTH (search depth of the ML bot in the matches, default 2), PYTHON.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-data/evaluations.csv}"
GAMES="${GAMES:-10}"
DEPTH="${DEPTH:-2}"
PYTHON="${PYTHON:-python}"

# Models trained from scratch by this repository, cheapest first.
TRAIN=(linear_baseline ridge_features mlp_features)
if [ "${TRAIN_CNN:-0}" = "1" ]; then TRAIN+=(cnn_resnet); fi

# Every model that has a saved file gets benchmarked (cnn_light is a frozen course artifact).
EVALUATE=(linear_baseline ridge_features mlp_features cnn_light cnn_resnet)

for model in "${TRAIN[@]}"; do
  echo "=== train ${model}"
  "$PYTHON" -m chess_engine.training.train --model "$model" --data "$DATA"
done

for model in "${EVALUATE[@]}"; do
  echo "=== evaluate ${model}"
  if [ "$GAMES" -gt 0 ]; then
    "$PYTHON" -m chess_engine.training.evaluate --model "$model" --data "$DATA" --strength --games "$GAMES" --depth "$DEPTH"
  else
    "$PYTHON" -m chess_engine.training.evaluate --model "$model" --data "$DATA"
  fi
done

"$PYTHON" scripts/plot_models.py
echo "Done. Review docs/training/MODELS_HISTORY.md and docs/img/, then commit src/chess_engine/models/ and docs/."
