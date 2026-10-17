---
tipo: decision
estado: aceptada
fecha: 2026-10-16
---
# 0003. Protocolo de la noche 1: presupuesto igual y margen de mejora

## Contexto
Elegir un ganador tras un entrenamiento corto e igual, y gastar luego el presupuesto grande en él, falla
por tres motivos: sesgo de margen (un modelo profundo aún mejora cuando se para), veredicto ruidoso (un
torneo de pocos cientos de partidas separa solo ±40 Elo) y decisiones encadenadas (la representación
elegida con una familia se reusa en otras).

## Decisión
- Todos los candidatos entrenan con el mismo flujo de posiciones, la misma representación por defecto
  (probabilidad de victoria, `k = 0.00368208`, punto de vista del que mueve) y el mismo número de posiciones.
- Se registra la pérdida de validación tras cada pasada. La pendiente final es el **margen de mejora**
  (columna `headroom`).
- Orden: precisión, luego profundidad y acuerdo de búsqueda a tiempo fijo, y al final el torneo.
- Noche 2: contendientes = los que solapan intervalo con el líder más el de mayor margen.

## Alternativas descartadas
- Elegir un ganador tras la noche 1: sesgado por el margen de cada modelo.

## Consecuencias
Los resultados de la noche 1 incluyen una columna de margen y una de coste (plantilla en el plan de la noche 1, fuera del repo).
