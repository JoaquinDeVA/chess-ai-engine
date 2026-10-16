# Training tasks

Checklist to train, judge and compare the evaluation models, and to refresh `MODELS_HISTORY.md`,
`TRADEOFF.md` and `TOURNAMENT.md`. The design is in
[docs/superpowers/specs/2026-10-16-training-and-evaluation-design.md](../superpowers/specs/2026-10-16-training-and-evaluation-design.md).

**Rule:** one model per *approach*. Retraining an approach overwrites its card
(`models/cards/<name>.json`); a new name is only added for a genuinely different approach. The four
older models (`ridge_features`, `mlp_features`, `cnn_light`, `cnn_resnet`) are only measured: they are
not retrained and do not play.

## Run everything (unattended)

- [ ] Raw Lichess evaluations in `data/lichess/raw/` (sharded Parquet). Either a local dump:
  `python scripts/fetch_lichess_raw.py --input data/lichess/lichess_db_eval.jsonl.zst`
  or streamed: `python scripts/fetch_lichess_raw.py --url <lichess eval database URL>` (public, CC0).
  Use `--rows` for a trial.
- [ ] `EFFORT=40 bash scripts/run_all.sh` (effort 0-100, see `scripts/run_all.sh`). The steps are:
  1. sidecars (split, quiet flag, side to move) for every raw shard (`scripts/prepare_dataset.py`);
  2. train `material_linear`, `pst_tapered` and `pst_terms` at `EFFORT`;
  3. evaluate every model on the fixed test sample: accuracy and cost;
  4. profile the depth against accuracy at 1, 3 and 10 s per move (`TRADEOFF.md`, plot);
  5. round-robin tournament for `HOURS` (default 10) (`TOURNAMENT.md`);
  6. regenerate the table in `MODELS_HISTORY.md`.
  Effort 100 for the three models may take a night; the tournament is the longest step.
- [ ] Look at `MODELS_HISTORY.md`, `TRADEOFF.md`, `TOURNAMENT.md` and `docs/img/depth_vs_accuracy.png`,
  then commit the cards, artifacts and docs that changed.

## Single model

- [ ] Follow `docs/ADDING_A_MODEL.md` (a feature set and a model class, registered by name).
- [ ] `python -m chess_engine.training.train --model <name> --effort <0-100>`
- [ ] `python -m chess_engine.training.evaluate --model <name>`
- [ ] `python scripts/profile_tradeoff.py --players <name>,h2,h3 --seconds 1,3,10`
- [ ] `python scripts/run_tournament.py --players <name>,h3,...` (must include `h3`, the Elo anchor)
- [ ] `python -m chess_engine.training.report` and commit.

## Optional

- [ ] Label your own positions: `python scripts/label_with_stockfish.py --random 50000 --depth 12`.
- [ ] Remove a model that loses the comparison: its artifact, card, config and code if unused.
