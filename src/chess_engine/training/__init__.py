"""Reproducible training and benchmarking of the evaluation models.

Typical flow::

    python scripts/download_lichess_evals.py --output data/evaluations.csv --rows 1000000
    python -m chess_engine.training.train --model mlp_features --data data/evaluations.csv
    python -m chess_engine.comparison.evaluate --model mlp_features --data data/evaluations.csv
"""
