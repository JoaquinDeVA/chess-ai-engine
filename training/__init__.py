"""Reproducible training and benchmarking of the evaluation models.

Typical flow::

    python scripts/download_lichess_evals.py --output data/evaluations.csv --rows 1000000
    python -m training.train --model mlp_features --data data/evaluations.csv
    python -m training.evaluate --model mlp_features --data data/evaluations.csv
"""
