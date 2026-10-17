---
tipo: modulo
source:
  - src/chess_engine/training/sidecar.py
  - src/chess_engine/training/lichess_raw.py
commit: 27a9f12fe844868038da03d009bd66f4ed3e7e6e
last_sync: 2026-10-17
---
# Muestreo y split de datos

## Propósito
Convertir la base de Lichess en fragmentos Parquet crudos, añadir a cada uno un fichero auxiliar (split,
marca de posición quieta, turno) y muestrear filas de un split sin cargar todo en memoria.

## Interfaces
- `lichess_raw.parse_lines`, `parse_chunk`, `write_shard`, `ShardWriter`: conversión de líneas JSON a
  `RAW_SCHEMA` (`fen`, `cp`, `mate`, `depth`, `knodes`, `best_move`, `n_evals`, `n_pvs`).
- `sidecar.build_sidecar(shard, out_dir)`: un auxiliar por fragmento, con las mismas filas y grupos.
- `sidecar.load_sample(raw_dir, sidecar_dir, split, n, seed, quiet_only, mate_keep)`: muestra uniforme por
  fila con memoria acotada a un fragmento. Devuelve un `DataFrame`.
- `sidecar.iter_batches(...)`: flujo de lotes para entrenar.

## Dependencias
Lo usan `training/train.py`, `training/effort.py` y `scripts/data/prepare_dataset.py`. Lee lo que escribe
`scripts/data/fetch_lichess_raw.py`.

## Por qué es así
- El split sale de un hash CRC32 del FEN con sal, así una posición siempre cae en el mismo split aunque
  cambien los fragmentos.
- `mate_keep` también usa CRC32: el subconjunto de mates es estable entre ejecuciones.
- Con `mate_keep < 1` el entrenamiento se niega a sobrescribir el modelo sin `--tag`: el modelo sin
  submuestrear no debe cambiar por accidente.
- Los fragmentos se escriben a `.tmp` y se renombran al final: un fragmento existe completo o no existe.
- El orden de la base no es aleatorio (ver `externos/lichess-evaluation-database.md`); el muestreo por fila
  lo compensa, no las primeras filas.
