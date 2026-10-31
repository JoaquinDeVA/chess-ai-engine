"""Data preparation and training of the evaluation models.

The linear models are trained from the raw Lichess evaluations, stored as sharded Parquet
(``data/lichess/raw``) with sidecars (split, quiet flag, side to move) next to them.
Judging and comparing the trained models is in
:mod:`chess_lab.measure`.

Typical flow::

    python src/chess_lab/scripts/data/build_raw_shards.py --input data/lichess/lichess_db_eval.jsonl.zst
    python src/chess_lab/scripts/data/build_sidecars.py
    python -m chess_lab.training.train --model linear_pst --effort 40
    python -m chess_lab.measure.evaluate --model linear_pst

"""
