"""Data preparation and training of the evaluation models.

The linear models are trained from the raw Lichess evaluations, stored as sharded Parquet
(``data/lichess/raw``) with sidecars (split, quiet flag, side to move) next to them. The older
CSV-era models train from a CSV. Judging and comparing the trained models is in
:mod:`chess_engine.comparison`.

Typical flow::

    python scripts/data/fetch_lichess_raw.py --input data/lichess/lichess_db_eval.jsonl.zst
    python scripts/data/prepare_dataset.py
    python -m chess_engine.training.train --model linear_pst --effort 40
    python -m chess_engine.comparison.evaluate --model linear_pst

CSV flow for the older models::

    python scripts/data/download_lichess_evals.py --output data/evaluations.csv --rows 1000000
    python -m chess_engine.training.train --model mlp_features --data data/evaluations.csv
    python -m chess_engine.comparison.evaluate --model mlp_features --data data/evaluations.csv
"""
