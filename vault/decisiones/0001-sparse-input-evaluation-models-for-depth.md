---
tipo: decision
estado: aceptada
fecha: 2026-10-16
---
# 0001. Modelos de evaluación con entrada dispersa para ganar profundidad

## Contexto
El objetivo es más profundidad de búsqueda con el mismo tiempo por jugada. Las redes convolucionales
tardan unos 13 ms por evaluación, unas 100 veces más que cualquier candidato lineal.

## Decisión
Todos los candidatos son modelos de entrada dispersa: una posición activa como mucho ~32 (o ~64)
características y la puntuación es función de la suma de las filas de pesos activas. Así son ligeros
y un movimiento solo cambia unas pocas características (base de la actualización incremental).

## Alternativas descartadas
- Redes convolucionales: solo quedan como registro de precisión.
- Árboles y gradient boosting: más precisión por característica, pero sin actualización incremental y
  ~10 veces más lentos por evaluación en Python.
- Ajuste por resultados de partidas (Texel con resultados): la base de Lichess no tiene resultados.
- Optimización por autojuego (SPSA/CLOP): requiere decenas de miles de partidas por paso.

## Consecuencias
Las familias son `pst_tapered`, `pst_terms`, `pst_kb` y `nn4` (ver `docs/training/SELECTION_PROTOCOL.md`).
Inferencia solo con NumPy, sin TensorFlow al jugar.
