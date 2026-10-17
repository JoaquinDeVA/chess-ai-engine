---
tipo: glosario
---
# Glosario

Solo términos del negocio y del proyecto, no conceptos generales de programación. Un término por entrada.

**Puntuación esperada**: probabilidad de victoria del bando que mueve, en [0,01, 0,99], derivada de la
evaluación en centipeones con la curva de Lichess (`k = 0.00368208`). Es lo que predicen los modelos.
_Evitar_: «evaluación» a secas (no se sabe si es centipeones o puntuación).

**Centipeones (cp)**: evaluación de Stockfish en cientos de un peón; positivo favorece a las blancas en la base de Lichess.

**Mate (mate)**: distancia al mate en la base de Lichess; positivo significa que mate blancas.

**Posición quieta (quiet)**: posición sin captura ni promoción en la mejor jugada y sin jaque. Es la marca del
sidecar; la búsqueda no tiene quiescencia, así que las no quietas son el punto débil a vigilar.

**Punto de vista**: de quién es la evaluación. Los modelos predicen desde el que mueve; el dato crudo viene de blancas.

**Split**: partición train, validación o test, asignada por hash CRC32 del FEN con sal. Es estable entre ejecuciones.

**Sidecar**: fichero Parquet junto a cada fragmento crudo con split, marca de quieta y turno.

**Fragmento (shard)**: un Parquet crudo de 1.000.000 de líneas de la base de Lichess.

**mate_keep**: fracción de posiciones con mate que se conserva al entrenar (1,0 por defecto). Los modelos con
`@mates5` o `@mates10` usan 0,05 o 0,10 y se etiquetan con `--tag`.

**Headroom (margen de mejora)**: pérdida de validación que todavía cae al final del entrenamiento, en %. Mide si
un modelo ha convergido. Umbrales en el plan de la noche 1 (fuera del repo).

**Effort (esfuerzo)**: nivel de entrenamiento (0-100) que fija cuántas posiciones se usan. Los candidatos entrenan con 30.

**Ancla (h1)**: referencia clásica con pesos ajustados (`evaluation/heuristics.py`). Los Elo del torneo son relativos a ella.

**h1**: única heurística clásica que queda. Usa las características de la antigua h3 con pesos ajustados a Stockfish por mínimos cuadrados (`h1_weights.json`, R² fuera de muestra 0,233 frente a 0,074 con los pesos originales).

**Familias de modelos**: `linear_material` (`material_linear`) (referencia), `linear_pst` (`pst_tapered`) (principal lineal), `linear_pst_terms` (`pst_terms`) (con términos
clásicos), `linear_king_bucket` (`pst_kb`) (con cubos por rey), `net4_pst` (`nn4`) (red de 4 capas). Ver [[0001-sparse-input-evaluation-models-for-depth]].

**Modelo registro de precisión**: modelo antiguo (CNN, MLP) que se mide pero no se entrena ni juega.
_Evitar_: «modelo viejo», que no dice si se usa.

**Profundidad completada**: última iteración de profundización terminada dentro del tiempo. Una iteración
cortada no cuenta.

**Noche 1 / noche 2**: primera ronda de entrenamiento, comparación y torneo con presupuesto igual; segunda ronda
con los contendientes elegidos. Ver [[0003-night-1-protocol-equal-budget-and-headroom]].
