#!/usr/bin/env bash
# The reproducible pipeline: sidecars, training, evaluation and the results table.
# Games are not played from here: they are played by the UCI engine and fastchess.
#
# Steps (STEPS, in this order): sidecars train evaluate report.
# Each step runs once and writes $STATE_DIR/<step>.done; a step with a .done file is skipped
# unless FORCE=1. A failing step is logged and stops the pipeline (non-zero exit), except model
# training: a failed model is logged and the loop continues. If the failed model is net4_pst its
# artifact and card are deleted ("DISCARDED net4_pst"). Per-model training logs go to
# $STATE_DIR/logs/. DRY_RUN=1 prints every command (one per line, prefixed "+ ") and writes nothing.
#
# h1 is fixed (src/chess_engine/models/h1_weights.json): changing it would change the anchor, so every result
# measured against the old h1 would stop being comparable.
#
# Variables (defaults): RAW (data/lichess/raw), SIDECAR (data/lichess/sidecar), EFFORT (30),
# AGREEMENT_N (20000), MODELS ("linear_material linear_pst linear_king_bucket linear_pst_terms net4_pst"),
# STEPS (all of the above), FORCE (0), DRY_RUN (0), STATE_DIR (data/pipeline),
# PYTHON (.venv/bin/python if it exists, else python).
#
#   Dry run:   DRY_RUN=1 bash src/chess_lab/scripts/run_full_pipeline.sh
#   Real run:  nohup setsid bash src/chess_lab/scripts/run_full_pipeline.sh > data/pipeline.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

RAW="${RAW:-data/lichess/raw}"
SIDECAR="${SIDECAR:-data/lichess/sidecar}"
EFFORT="${EFFORT:-30}"
AGREEMENT_N="${AGREEMENT_N:-20000}"
MODELS="${MODELS:-linear_material linear_pst linear_king_bucket linear_pst_terms net4_pst}"
STEPS="${STEPS:-sidecars train evaluate report}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
STATE_DIR="${STATE_DIR:-data/pipeline}"
if [ -z "${PYTHON:-}" ]; then
  if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python; fi
fi

ARTIFACTS=src/chess_engine/models/artifacts
CARDS=src/chess_lab/models/cards

log() { echo "=== $*"; }

# Run a command, or only print it when DRY_RUN=1.
run() {
  if [ "$DRY_RUN" = 1 ]; then echo "+ $*"; else "$@"; fi
}

registered() {
  "$PYTHON" -c "import sys; from chess_lab.models.definitions import available_models as a; sys.exit(0 if '$1' in a() else 1)" 2>/dev/null
}

# Models of MODELS that are registered and (outside a dry run) have a trained artifact.
present_models() {
  local m out=()
  for m in $MODELS; do
    if [ "$DRY_RUN" = 1 ]; then out+=("$m")
    elif registered "$m" && [ -f "$ARTIFACTS/$m.npz" ]; then out+=("$m"); fi
  done
  echo "${out[*]:-}"
}

join() { local IFS=,; echo "$*"; }

step_sidecars() {
  run "$PYTHON" src/chess_lab/scripts/data/build_sidecars.py --raw "$RAW" --out "$SIDECAR"
}

step_train() {
  local model
  [ "$DRY_RUN" = 1 ] || mkdir -p "$STATE_DIR/logs"
  for model in $MODELS; do
    if [ "$DRY_RUN" != 1 ] && ! registered "$model"; then log "skip train $model: not registered"; continue; fi
    log "train $model"
    if [ "$DRY_RUN" = 1 ]; then
      echo "+ $PYTHON -m chess_lab.training.train --model $model --effort $EFFORT --raw $RAW --sidecar $SIDECAR"
    elif ! "$PYTHON" -m chess_lab.training.train --model "$model" --effort "$EFFORT" \
        --raw "$RAW" --sidecar "$SIDECAR" > "$STATE_DIR/logs/train_$model.log" 2>&1; then
      log "FAILED train $model (log: $STATE_DIR/logs/train_$model.log); the other models continue"
      if [ "$model" = net4_pst ]; then
        rm -f "$ARTIFACTS/net4_pst.npz" "$CARDS/net4_pst.json"
        log "DISCARDED net4_pst"
      fi
    fi
  done
}

step_evaluate() {
  local model
  for model in $(present_models) h1; do
    log "evaluate $model"
    run "$PYTHON" -m chess_lab.measure.evaluate --model "$model" \
      --raw "$RAW" --sidecar "$SIDECAR" --agreement-n "$AGREEMENT_N"
  done
}

step_report() {
  run "$PYTHON" -m chess_lab.measure.report
}

[ "$DRY_RUN" = 1 ] || mkdir -p "$STATE_DIR"
for step in $STEPS; do
  done_file="$STATE_DIR/$step.done"
  if [ "$FORCE" != 1 ] && [ -f "$done_file" ]; then log "skip $step: done"; continue; fi
  log "$step"
  if ! "step_$step"; then
    log "FAILED $step: pipeline stopped"
    exit 1
  fi
  [ "$DRY_RUN" = 1 ] || touch "$done_file"
done
log "Done. Review docs/RESULTS.md, data/reports/ and the cards in $CARDS and commit."
