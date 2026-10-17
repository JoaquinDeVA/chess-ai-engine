#!/usr/bin/env bash
# Train, judge and compare the evaluation models: sidecars, the three linear models at EFFORT,
# evaluation of every model, depth-accuracy profile, round-robin tournament, models table.
#
#   EFFORT=40 bash scripts/run_all.sh
#
# Variables (defaults): RAW (data/lichess/raw), SIDECAR (data/lichess/sidecar), EFFORT (30),
# EFFORT_NN (30, networks), AGREEMENT_N (20000, best-move sample rows), HOURS (4, tournament
# wall-clock budget), BASE (60) and INCREMENT (1) (tournament clock), PROFILE_SECONDS (1,3),
# PROFILE_POSITIONS (100), PLAYERS (tournament players, anchored at h1), PYTHON.
# Models not yet registered are skipped with a line in the log (the runbook says so too).
set -euo pipefail
cd "$(dirname "$0")/.."

RAW="${RAW:-data/lichess/raw}"
SIDECAR="${SIDECAR:-data/lichess/sidecar}"
EFFORT="${EFFORT:-30}"
EFFORT_NN="${EFFORT_NN:-30}"
AGREEMENT_N="${AGREEMENT_N:-20000}"
HOURS="${HOURS:-4}"
BASE="${BASE:-60}"
INCREMENT="${INCREMENT:-1}"
PROFILE_SECONDS="${PROFILE_SECONDS:-1,3}"
PROFILE_POSITIONS="${PROFILE_POSITIONS:-100}"
PLAYERS="${PLAYERS:-linear_pst,linear_pst_terms,linear_king_bucket,net4_pst,h1}"
PYTHON="${PYTHON:-python}"

# Registered models only (a model whose code is not merged yet is skipped, with a log line).
registered() {
  "$PYTHON" -c "import sys; from chess_engine.models import available_models as a; sys.exit(0 if '$1' in a() else 1)" 2>/dev/null
}

# Models trained from the raw shards (order matters: cheapest first), and every model evaluated
# (the older ones are measured only, never retrained).
TRAINED=(linear_material linear_pst linear_pst_terms linear_king_bucket net4_pst)
EVALUATED=(linear_material linear_pst linear_pst_terms linear_king_bucket net4_pst mlp_features cnn_resnet)

echo "=== sidecars"
"$PYTHON" scripts/data/prepare_dataset.py --raw "$RAW" --out "$SIDECAR"

for model in "${TRAINED[@]}"; do
  if ! registered "$model"; then echo "=== skip train ${model}: not registered"; continue; fi
  effort="$EFFORT"; case "$model" in nn*) effort="$EFFORT_NN" ;; esac
  echo "=== train ${model} (effort ${effort})"
  "$PYTHON" -m chess_engine.training.train --model "$model" --effort "$effort" \
    --raw "$RAW" --sidecar "$SIDECAR" || echo "=== FAILED train ${model}: the other models continue"
done

for model in "${EVALUATED[@]}"; do
  if ! registered "$model"; then echo "=== skip evaluate ${model}: not registered"; continue; fi
  echo "=== evaluate ${model}"
  "$PYTHON" -m chess_engine.comparison.evaluate --model "$model" \
    --raw "$RAW" --sidecar "$SIDECAR" --agreement-n "$AGREEMENT_N" \
    || echo "=== FAILED evaluate ${model}"
done

echo "=== profile (depth against accuracy)"
PROFILE_PLAYERS="$(echo "$PLAYERS" | tr ',' '\n' | while read -r p; do if [ "$p" = h1 ] || registered "$p"; then echo "$p"; fi; done | paste -sd, -)"
"$PYTHON" scripts/bench/profile_tradeoff.py --players "$PROFILE_PLAYERS" --seconds "$PROFILE_SECONDS" \
  --positions "$PROFILE_POSITIONS" --parallel 1 --raw "$RAW" --sidecar "$SIDECAR"

WEIGHTS="src/chess_engine/evaluation/h1_weights.json"
if [ ! -f "$WEIGHTS" ] || [ -n "$(find "$SIDECAR" -name '*.sidecar.parquet' -newer "$WEIGHTS" -print -quit)" ]; then
  echo "=== fit h1"
  "$PYTHON" scripts/data/fit_h1.py --raw "$RAW" --sidecar "$SIDECAR" --positions 300000 --jobs 4 \
    --out "$WEIGHTS"
else
  echo "=== fit h1 skipped (weights newer than the sidecars)"
fi

echo "=== tournament (${HOURS} h)"
TOURNAMENT_PLAYERS="$(echo "$PLAYERS" | tr ',' '\n' | while read -r p; do if [ "$p" = h1 ] || registered "$p"; then echo "$p"; fi; done | paste -sd, -)"
"$PYTHON" scripts/bench/run_tournament.py --players "$TOURNAMENT_PLAYERS" --anchor h1 \
  --base "$BASE" --increment "$INCREMENT" --hours "$HOURS"

echo "=== report"
"$PYTHON" -m chess_engine.comparison.report
echo "Done. Review docs/training/ (MODELS_HISTORY.md, TRADEOFF.md, TOURNAMENT.md) and commit."
