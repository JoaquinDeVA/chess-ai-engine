# Runbook — night 1 (train, measure, compare)

For an executor that runs commands and reads logs. **You do not design, you do not decide, you do not
edit code.** You launch each step, check that it is healthy, collect numbers from the files named here,
write them into [RESULTS.md](RESULTS.md) with the interpretation rules below, and report. The strategy
behind the steps is in [SELECTION_PROTOCOL.md](SELECTION_PROTOCOL.md).

Repository: `/mnt/Data/09-Proyectos/07-ChessAIEngine`. Always `cd` there first and use `.venv/bin/python`
(never the system Python). Export `TF_CPP_MIN_LOG_LEVEL=3` in every shell. Replies to the user in Spanish.

## Rules (read twice)

1. **Stop and report** (a message to the supervisor with the step name and the last 30 lines of the log)
   whenever: a command exits non-zero, a log shows `Traceback`, an estimate in this runbook is exceeded by
   more than 2x, a file this runbook says must exist is missing, or you are about to do something this
   runbook does not list. Retry a failed command **once** only, and only if the cause was obviously
   transient (network, a killed process).
2. **Never** delete files under `data/`, `src/`, `docs/` or `.git`; never `git reset`, `git stash`,
   `git checkout --`, `git push`; never change code or configuration; never lower an effort, a
   position count or a time limit to "make it fit". Commits are allowed only where a step says so, with
   a plain message and **no** `Co-Authored-By` / "Generated with" lines.
3. **Never use `pgrep -f` with a pattern that also appears in your own command line**: it matches itself and
   waits forever. Use the files named below to know whether a job finished.
4. Long jobs run **detached** with `nohup setsid ... > <log> 2>&1 &`; you then poll the log with
   `tail -n 5 <log>`, never more often than every 5 minutes (use `sleep 300` inside one command with a
   `timeout` of at most 10 minutes, then check). Nothing here needs a GPU.
5. CPU is shared: run **one** heavy job at a time, in the order of the steps.
6. Decisions are marked **STOP-DECISION**: write what you collected, report, and wait.

## Step 0 — Pre-flight

```bash
cd /mnt/Data/09-Proyectos/07-ChessAIEngine && export TF_CPP_MIN_LOG_LEVEL=3
git log --oneline | head -12; git status --short | head
ls data/lichess/raw/*.parquet | wc -l            # expected: 126
ls data/lichess/sidecar/*.sidecar.parquet 2>/dev/null | wc -l
ls data/lichess/sidecar/*.tmp 2>/dev/null | wc -l
.venv/bin/python -c "from chess_engine.models import available_models; print(available_models())"
df -h /mnt/Data | tail -1; free -g | head -2
```

Expected: 126 raw shards (about 124.9 million positions: the first 124.9 M lines of the Lichess evaluation
database; the file is **not** randomly ordered, early lines are deeper and have more mates — note this
in RESULTS.md "Data"); the model list contains at least `material_linear`, `pst_tapered`, `pst_terms`.
`pst_kb` and `nn4` appear when their code lands (the supervisor tells you, or you see the commits
`feat: slotted and king-bucket features...` and `feat: sparse-input neural evaluation model...` in
`git log`). If a model is not registered yet, skip its steps and mark it "pending code" in RESULTS.md;
rerun those steps later.

## Step 1 — Sidecars (split, quiet flag) — the supervisor already launched them

Log: `data/runs/night1/sidecars.log`. Done when `ls data/lichess/sidecar/*.sidecar.parquet | wc -l` is
**126** and there are **0** `*.tmp` files in that directory.

```bash
until [ "$(ls data/lichess/sidecar/*.sidecar.parquet 2>/dev/null | wc -l)" -ge 126 ] \
  && [ "$(ls data/lichess/sidecar/*.tmp 2>/dev/null | wc -l)" -eq 0 ]; do sleep 300; done; echo sidecars-ready
```

(Use `timeout 600` around it and repeat; an estimate is about 2 hours in total. If the log has not grown for
30 minutes, stop and report.) If the process died before 126, relaunch once (it skips finished shards):

```bash
nohup setsid .venv/bin/python scripts/data/prepare_dataset.py --raw data/lichess/raw --out data/lichess/sidecar --jobs 4 \
  > data/runs/night1/sidecars2.log 2>&1 &
```

**COLLECT** → RESULTS.md "Data": run

```bash
.venv/bin/python - <<'EOF'
import glob, pyarrow.parquet as pq, numpy as np
n = [0, 0, 0]; q = 0; total = 0; white = 0
for f in sorted(glob.glob("data/lichess/sidecar/*.sidecar.parquet")):
    t = pq.read_table(f, columns=["split", "quiet", "white_to_move"])
    s = t["split"].to_numpy()
    for k in range(3): n[k] += int((s == k).sum())
    q += int(t["quiet"].to_numpy().sum()); total += len(s); white += int(t["white_to_move"].to_numpy().sum())
print("rows", total, "train/val/test", n, "quiet share", round(q / total, 3), "white to move", round(white / total, 3))
EOF
```

Expected: train about 80%, validation about 10%, test about 10%.

## Step 2 — Fit `h3t`, the tuned classical reference  (needs the commit `feat: h3t...`)

```bash
nohup setsid .venv/bin/python scripts/data/fit_h3t.py --raw data/lichess/raw --sidecar data/lichess/sidecar \
  --positions 300000 --jobs 4 > data/runs/night1/fit_h3t.log 2>&1 &
```

Done when the log prints both R² values and `src/chess_engine/evaluation/h3t_weights.json` exists (about
10–25 minutes). **COLLECT** → RESULTS.md "Reference h3t": `r2` of the fitted weights, `r2_h3` of the original
H3 weights, the six fitted weights (read the JSON). Rule: the fitted R² must be **greater** than `r2_h3`; if
not, STOP and report. Commit the weights file:
`git add src/chess_engine/evaluation/h3t_weights.json && git commit -m "data: fitted h3t weights"`.

## Step 3 — Train the models (one at a time, in this order)

Models: `material_linear`, `pst_tapered`, `pst_terms`, `pst_kb`, `nn4` (skip the ones not registered yet).

```bash
M=pst_tapered   # repeat for each model
nohup setsid .venv/bin/python -m chess_engine.training.train --model $M --effort 30 \
  > data/runs/night1/train_$M.log 2>&1 &
```

The command prints the configuration and an **estimated duration** first. Rules: if the estimate for any
model is longer than **90 minutes**, STOP-DECISION (do not lower the effort yourself). Poll the log; it is
done when the last line starts with `Saved` and the card `src/chess_engine/models/cards/$M.json` has
`training.effort == 30`.

**COLLECT** (per model, from the card `training` section; use
`.venv/bin/python -c "import json;t=json.load(open('src/chess_engine/models/cards/$M.json'))['training'];print({k:t.get(k) for k in ['effort','train_samples','seconds','epochs','best_epoch','val_losses','train_losses']})"`)
→ RESULTS.md "Training":

| model | parameters | positions | seconds | estimate printed | epochs | best epoch | val loss per epoch | headroom |

`headroom` = `val_losses[-2] - val_losses[-1]` divided by `val_losses[-1]`, as a percentage (if only one
epoch ran, write "n/a"). **Rule of interpretation:** headroom above 1% = "still improving: needs more
training"; below 0.2% = "converged"; between = "nearly converged". If `best_epoch` is smaller than `epochs - 2`
write "stopped by patience (saturated)".

## Step 4 — Evaluate every model (new and old)

```bash
for M in material_linear pst_tapered pst_terms pst_kb nn4; do
  .venv/bin/python -m chess_engine.comparison.evaluate --model $M --agreement-n 20000 \
    > data/runs/night1/eval_$M.log 2>&1 || echo "FAILED $M"
done
for M in ridge_features mlp_features; do
  .venv/bin/python -m chess_engine.comparison.evaluate --model $M --agreement-n 5000 \
    > data/runs/night1/eval_$M.log 2>&1 || echo "FAILED $M"
done
for M in cnn_light cnn_resnet; do
  .venv/bin/python -m chess_engine.comparison.evaluate --model $M --agreement-n 1000 \
    > data/runs/night1/eval_$M.log 2>&1 || echo "FAILED $M"
done
```

(Skip models not registered. Run as a detached script with `nohup setsid bash -c '...'` and poll if it
exceeds 10 minutes; the CNNs may take 10–30 minutes each — if one exceeds 45 minutes, kill only that
process with `kill <pid>` and report.) **COLLECT** from each card `accuracy` and `cost` sections →
RESULTS.md "Accuracy": Spearman and MAE (all / quiet / non_quiet), best-move agreement (all / non_quiet),
microseconds per evaluation. **Interpretation rules:** a model is "more accurate" than another only if its
Spearman (all) is larger by at least 0.01; always mention the non_quiet slice (the search has no quiescence,
tactical positions matter); quote the agreement sample size used (`agreement_n`).

## Step 5 — Depth against accuracy (profile)

```bash
nohup setsid .venv/bin/python scripts/bench/profile_tradeoff.py \
  --players pst_tapered,pst_terms,pst_kb,nn4,material_linear,h3t --seconds 1,3 --positions 100 --parallel 1 \
  > data/runs/night1/profile.log 2>&1 &
```

(Drop unregistered models from `--players`. Estimate: 100 positions x (1+3) s x players = 6 players ≈ 40
minutes.) It writes `docs/training/TRADEOFF.md`, `docs/training/tradeoff.json`, `docs/img/depth_vs_accuracy.png`
and the `tradeoff` card sections. If a player is listed under SKIPPED, report it. **COLLECT** →
RESULTS.md "Depth": mean completed depth at 1 s and 3 s per model and the search agreement. **Rule:** the
trade-off is read as "extra depth per second of thinking vs accuracy": write for each model the depth gained or
lost relative to `pst_tapered` and the accuracy (Spearman) gained or lost; do not conclude which is better
(that is the tournament's job).

## Step 6 — Tournament (night 1)

```bash
nohup setsid .venv/bin/python scripts/bench/run_tournament.py \
  --players pst_tapered,pst_terms,pst_kb,nn4,h3t --anchor h3t --base 60 --increment 1 --workers 5 --hours 4 \
  --out data/tournament/games.jsonl > data/runs/night1/tournament.log 2>&1 &
```

(Drop unregistered models from `--players`.) It first calibrates, then plays for 4 hours; the games file is
resumable: if the process dies, run **exactly the same command** again. Done when the log ends with the
report line and `docs/training/TOURNAMENT.md` exists. **COLLECT** → RESULTS.md "Tournament": the cross
table is generated; copy the ranking (score %, Elo ± interval, games per pair). **Interpretation rules:**
two models are "different" only if the Elo difference is larger than the square root of the sum of the squares
of their intervals; otherwise write "indistinguishable". Report the number of games per pair and the count
of crashed/error games (lines with `"error"` in the games file).

## Step 7 — Report and commit

```bash
.venv/bin/python -m chess_engine.comparison.report
git status --short
git add docs/training/*.md docs/training/*.json docs/img/depth_vs_accuracy.png src/chess_engine/models/cards/*.json \
  src/chess_engine/models/artifacts/*.npz src/chess_engine/models/artifacts/*.keras src/chess_engine/models/artifacts/*.pkl 2>/dev/null
git commit -m "results: night 1 training, profile and tournament"
```

Check that `git status --short` afterwards shows nothing important unstaged (data under `data/` is ignored
on purpose). Do not push.

## Step 8 — Findings and hand-over

Fill RESULTS.md "Findings" with at most 8 bullets, each with a number: who is most accurate, who reaches the
most depth, the ranking with intervals, headroom of each model, any anomaly (a model that failed, a
huge cost, an unusual log). Then **STOP-DECISION**: send the supervisor RESULTS.md plus the question
"which models go to night 2?" — you do not choose.

## Night 2 (only when the supervisor says so)

Fine-tuning of the contenders (higher `--effort`), the representation bake-off
(`scripts/bench/bakeoff_representation.py`) and the mate-undersampled variants
(`train --model <best> --mate-keep 0.05 --tag mates5`, same with `0.10` / `mates10`; then a tournament of
`<best>,<best>@mates5,<best>@mates10` with `--anchor <best>`) are described in
[SELECTION_PROTOCOL.md](SELECTION_PROTOCOL.md); the supervisor will give the exact commands.
