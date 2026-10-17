---
tipo: decision
estado: aceptada
fecha: 2026-10-16
---
# 0002. No usar la red de 8 capas (nn8)

## Contexto
Se diseñó una red de 8 capas con entradas por cubo de rey (`pst_kb` → 384 → 128 → 64 → 64 → 32 → 32 →
16 → 1), de unos 4 millones de parámetros.

## Decisión
Se descarta. Es demasiado pesada para una CPU: unos 20 veces los parámetros de `nn4`, entrenamiento más
lento y ~2 veces más lenta por evaluación, lo que cuesta profundidad de búsqueda. Fue decisión del usuario.

## Alternativas descartadas
- Mantener nn8 en el torneo: consume el presupuesto de la noche sin evidencia de que compense.
- Usar nn4 (~206.000 parámetros) como la red de 4 capas: es la alternativa elegida.

## Consecuencias
- Queda como opción de la noche 2 solo si `nn4` supera claramente a `pst_tapered` y todavía tiene margen.
- Se quitó de `run_all.sh`, de los jugadores por defecto del torneo y de los tests.
