#!/usr/bin/env bash
# Train, judge and compare the evaluation models: sidecars, the three linear models at EFFORT,
# evaluation of every model, depth-accuracy profile, round-robin tournament, models table.
#
#   EFFORT=40 bash scripts/run_all.sh
#
# Variables (defaults): RAW (data/lichess/raw), SIDECAR (data/lichess/sidecar), EFFORT (40),
# AGREEMENT_N (20000, best-move sample rows), HOURS (10, tournament wall-clock budget), PYTHON.
set -euo pipefail
cd "$(dirname "$0")/.."

RAW="${RAW:-data/lichess/raw}"
SIDECAR="${SIDECAR:-data/lichess/sidecar}"
EFFORT="${EFFORT:-40}"
AGREEMENT_N="${AGREEMENT_N:-20000}"
HOURS="${HOURS:-10}"
PYTHON="${PYTHON:-python}"

# Models trained from the raw shards, and every model evaluated (the older ones are measured only).
TRAINED=(material_linear pst_tapered pst_terms)
EVALUATED=(material_linear pst_tapered pst_terms ridge_features mlp_features cnn_light cnn_resnet)

echo "=== sidecars"
"$PYTHON" scripts/prepare_dataset.py --raw "$RAW" --out "$SIDECAR"

for model in "${TRAINED[@]}"; do
  echo "=== train ${model} (effort ${EFFORT})"
  "$PYTHON" -m chess_engine.training.train --model "$model" --effort "$EFFORT" \
    --raw "$RAW" --sidecar "$SIDECAR"
done

for model in "${EVALUATED[@]}"; do
  echo "=== evaluate ${model}"
  "$PYTHON" -m chess_engine.comparison.evaluate --model "$model" \
    --raw "$RAW" --sidecar "$SIDECAR" --agreement-n "$AGREEMENT_N"
done

echo "=== profile (depth against accuracy)"
"$PYTHON" scripts/profile_tradeoff.py --raw "$RAW" --sidecar "$SIDECAR"

echo "=== tournament (${HOURS} h)"
"$PYTHON" scripts/run_tournament.py --hours "$HOURS"

echo "=== report"
"$PYTHON" -m chess_engine.comparison.report
echo "Done. Review docs/training/ (MODELS_HISTORY.md, TRADEOFF.md, TOURNAMENT.md) and commit."
