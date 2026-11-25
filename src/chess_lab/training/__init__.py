"""Training of the evaluation networks, from the data that ``chess_lab.datasets`` prepares.

Judging and comparing the trained models is in :mod:`chess_lab.measure`.

Typical flow::

    python -m chess_lab.datasets.build_raw_shards --input data/LichessPositions/lichess_db_eval.jsonl.zst
    python -m chess_lab.datasets.build_sidecars
    python -m chess_lab.training.train --model net_pst_threats_w64_both --effort 40
    python -m chess_lab.measure.evaluate --model net_pst_threats_w64_both

"""
