---
tipo: modulo
source:
  - src/chess_engine/search
commit: ecd13ed2a839873046207b2d2e950e84319de2c7
last_sync: 2026-10-17
---
# Búsqueda adversarial

## Propósito
Elegir la jugada con alfa-beta y profundización iterativa dentro de un presupuesto de tiempo. Es
independiente del juego: `GameState` y `GameProblem` son abstractos y el ajedrez los implementa encima de
python-chess.

## Interfaces
- `heuristic_alphabeta_iterative_deepening_search(game, state, eval_fn, cutoff_test, ..., time_limit)`:
  punto de entrada que usan los bots. Devuelve `(jugada, SearchStatistics)`.
- `advanced_heuristic_alphabeta_search(...)`: una profundidad, con tabla de transposición, killer moves y
  ordenación de jugadas por callback.
- `SearchStatistics.completed_depth`: profundidad de la última iteración terminada. Es la cifra que mide
  el trade-off de profundidad (`comparison/tradeoff.py`).
- `ChessGameState.apply_move` / `undo_move`: hacer y deshacer jugadas sobre el mismo tablero (no copiar).
- `ChessGameState.key()`: clave de la tabla de transposición. Incluye el reloj de jugadas de 50 y el número
  de jugada, porque la regla de 50 jugadas depende de ellos.

## Dependencias
Usa `evaluation/` y `models/` a través de `eval_fn`. No conoce modelos concretos. Lo usan `bots/` y
`engine/`.

## Por qué es así
- Las reglas de tablas (triple repetición y regla de 50 jugadas) siguen exactamente a
  `board.outcome(claim_draw=True)` de python-chess. Cambiar eso cambia los nodos buscados y las jugadas.
- Hacer y deshacer en lugar de copiar el tablero acelera sin cambiar jugadas ni nodos buscados. Por eso
  las profundidades medidas antes y después de la optimización son comparables.
- `completed_depth` cuenta solo iteraciones terminadas; una iteración cortada por tiempo no cuenta.
