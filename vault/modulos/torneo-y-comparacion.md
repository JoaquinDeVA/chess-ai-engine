---
tipo: modulo
source:
  - src/chess_engine/comparison
commit: 519dbfe50ce5ef01db568f2df8cffe87c2d13985
last_sync: 2026-10-17
---
# Torneo y comparación

## Propósito
Medir modelos y bots de tres formas: precisión frente a Stockfish (`judge`), profundidad frente a
precisión a tiempo fijo por jugada (`tradeoff`) y round-robin con Elo (`tournament`).

## Interfaces
- `judge.judge(model, test, agreement_n)`: Spearman, MAE y acuerdo de la mejor jugada.
- `tradeoff.profile(...)`, `run_profiles`, `record_cards`: profundidad completada a 1 s y 3 s.
- `tournament.run_round_robin(specs, openings, base, increment, workers, out)`: cada pareja juega cada
  apertura dos veces con colores cambiados. Escribe `games.jsonl` y es reanudable.
- `tournament.summarize(results, anchor)`: puntuación y Elo con el ancla fijada.
- `tournament.size_for_budget(hours, workers, n_players, seconds_per_game)`: partidas por pareja.

## Dependencias
Usa `training/` y `models/` solo para cargar modelos; `comparison/` nunca es importado por ellos. Lo usan
`scripts/bench/run_tournament.py`, `profile_tradeoff.py` y `scripts/run_all.sh`.

## Por qué es así
- El ancla es `h3t` (`ANCHOR`), la referencia clásica ajustada. Los Elo son relativos a ella, no absolutos.
- Cada apertura se juega con colores cambiados: sin eso la ventaja de blancas sesga la puntuación.
- Un juego que muere en `MAX_DEATHS_PER_GAME` = 2 pools rotos se registra como error y no se reintenta:
  evita bucles infinitos ante un fallo sistémico.
- El contador de reconstrucción del pool solo se reinicia con partidas jugadas, no con errores.
- Dos modelos son distintos solo si la diferencia de Elo supera la raíz de la suma de los cuadrados de sus
  intervalos (regla de `docs/training/RUNBOOK.md`).
