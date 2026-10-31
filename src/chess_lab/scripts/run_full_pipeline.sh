#!/usr/bin/env bash
# The reproducible pipeline: sidecars, training, evaluation, depth profile, three tournaments,
# quiescence comparison and the results table. It is also what produces the real results.
#
# Steps (STEPS, in this order): sidecars train evaluate profile tournaments quiescence report.
# Each step runs once and writes $STATE_DIR/<step>.done; a step with a .done file is skipped
# unless FORCE=1. A failing step is logged and stops the pipeline (non-zero exit), except model
# training: a failed model is logged and the loop continues. If the failed model is net4_pst its
# artifact and card are deleted ("DISCARDED net4_pst"). Per-model training logs go to
# $STATE_DIR/logs/. DRY_RUN=1 prints every command (one per line, prefixed "+ ") and writes nothing.
#
# Tournaments: A) 60+1 anchored at h1; B) 600+5 (the reference) anchored at h1; C) depth 3
# with no clock, without h1, anchored at linear_pst_terms.
#
# h1 is fixed (src/chess_engine/models/h1_weights.json): changing it would change the anchor, so every result
# measured against the old h1 would stop being comparable.
#
# Variables (defaults): RAW (data/lichess/raw), SIDECAR (data/lichess/sidecar), EFFORT (30),
# AGREEMENT_N (20000), PROFILE_SECONDS (1,3,10), PROFILE_POSITIONS (100), WORKERS (5),
# MODELS ("linear_material linear_pst linear_king_bucket linear_pst_terms net4_pst"),
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
PROFILE_SECONDS="${PROFILE_SECONDS:-1,3,10}"
PROFILE_POSITIONS="${PROFILE_POSITIONS:-100}"
WORKERS="${WORKERS:-5}"
# Estimator of the next iteration used by the tournament players. The games recorded in data/tournament
# were played with the engine's original rule (ratio_squared); the engine default has been another
# one since 2026-10-21. Keep the value to resume or reproduce those tournaments; set it to
# "geometric_growth" (or leave the engine default with TOURNAMENT_ESTIMATOR=default) for a fresh run.
TOURNAMENT_ESTIMATOR="${TOURNAMENT_ESTIMATOR:-ratio_squared}"
MODELS="${MODELS:-linear_material linear_pst linear_king_bucket linear_pst_terms net4_pst}"
STEPS="${STEPS:-sidecars train evaluate profile tournaments quiescence report}"
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

step_profile() {
  local players
  players="$(join $(present_models) h1)"
  run "$PYTHON" src/chess_lab/scripts/bench/profile_depth_vs_accuracy.py --players "$players" \
    --seconds "$PROFILE_SECONDS" --positions "$PROFILE_POSITIONS" --parallel 1 \
    --raw "$RAW" --sidecar "$SIDECAR"
}

estimator_args() {
  [ "$TOURNAMENT_ESTIMATOR" = default ] || echo "--estimator $TOURNAMENT_ESTIMATOR"
}

step_tournaments() {
  local with_h1 without_h1
  with_h1="$(join $(present_models) h1)"
  without_h1="$(join $(present_models))"
  [ "$DRY_RUN" = 1 ] || mkdir -p data/tournament data/reports
  # A) short control
  run "$PYTHON" -m chess_lab.tournaments.play_tournament --players "$with_h1" --anchor h1 \
    --base 60 --increment 1 --openings 10 --workers "$WORKERS" $(estimator_args) \
    --out data/tournament/short.jsonl --logs data/tournament/short \
    --report data/reports/TOURNAMENT_60+1.md
  # B) reference control
  run "$PYTHON" -m chess_lab.tournaments.play_tournament --players "$with_h1" --anchor h1 \
    --base 600 --increment 5 --openings 5 --workers "$WORKERS" $(estimator_args) \
    --out data/tournament/reference.jsonl --logs data/tournament/reference \
    --report data/reports/TOURNAMENT_600+5.md
  # C) equal depth 3, no h1
  run "$PYTHON" -m chess_lab.tournaments.play_tournament --players "$without_h1" --anchor linear_pst_terms \
    --max-depth 3 --base 600 --increment 0 --openings 10 --workers "$WORKERS" $(estimator_args) \
    --out data/tournament/depth3.jsonl --logs data/tournament/depth3 \
    --report data/reports/TOURNAMENT_depth3.md
}

step_quiescence() {
  run "$PYTHON" src/chess_lab/scripts/bench/compare_quiescence.py --players h1,linear_pst_terms \
    --raw "$RAW" --sidecar "$SIDECAR"
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
