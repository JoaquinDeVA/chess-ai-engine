# Modelos y resultados

> **Qué es:** Los datos de entrenamiento, la escala de esfuerzo y los resultados medidos de cada modelo, con el comando que reproduce cada número.
>
> **Cuándo leerla:** Para comparar modelos, preparar datos o repetir una medición.
>
> **Siguiente paso:** [Añadir un modelo](ADDING_A_MODEL.md)

Los números son de este proyecto (CPU, esfuerzo 15 salvo que se diga otra cosa). Los comandos se
lanzan desde la raíz del repositorio.

## Datos

**Origen.** Base pública de evaluaciones de Lichess, `lichess_db_eval.jsonl.zst`
(<https://database.lichess.org/lichess_db_eval.jsonl.zst>, CC0, 22.4 GB, 416,442,401 posiciones
evaluadas con Stockfish). Una línea JSON por posición: `fen` (sin contadores de jugadas) y `evals`
(varias evaluaciones de la misma posición, cada una con `pvs`, `knodes` y `depth`). El fichero no
está ordenado al azar: las posiciones más profundas y equilibradas vienen primero, así que las
primeras no son una muestra uniforme de la base.

**Lo que se usa:** los primeros 124,923,027 registros, en 126 shards Parquet
(`data/lichess/raw/part-NNNNN.parquet`, git-ignorados). Cada fila sale de la evaluación de mayor
`depth` (empate: mayor `knodes`):

| columna | tipo | significado |
|---|---|---|
| `fen` | string | posición, 4 campos |
| `cp` | int16, nulo | centipeones de `pvs[0]` (acotado a -20000..20000) |
| `mate` | int16, nulo | distancia al mate de `pvs[0]`; exactamente uno de `cp`/`mate` está definido |
| `depth` | int16 | profundidad de la evaluación elegida |
| `knodes` | int32 | miles de nodos de esa evaluación |
| `best_move` | string | primer movimiento (UCI) de `pvs[0]` |
| `n_evals`, `n_pvs` | uint8 | evaluaciones de la línea y PVs de la elegida |

**Convención de signo.** `cp` y `mate` están siempre **desde el punto de vista de las blancas**
(positivo = bien para blancas, `mate > 0` = mate de blancas), mueva quien mueva. Se detectó con
`chess_engine.training.lichess.detect_perspective` (correlación con el material en posiciones con
negras al turno). Al entrenar, el objetivo se pasa al punto de vista del bando que mueve
(`Representation`: `winprob`, `k = 0.00368208`, `side_to_move`).

**Sidecars.** `part-NNNNN.sidecar.parquet` junto a cada shard (mismas filas y grupos de filas):
partición (train, validación, test, por hash con semilla del FEN, `training.dataset.assign_split`),
posición tranquila (sin jaque y con mejor jugada que no es captura ni promoción) y bando que mueve.

| medida | valor |
|---|---|
| shards / filas | 126 / 124,923,027 |
| train / validación / test (80/10/10) | 99,935,706 / 12,496,155 / 12,491,166 |
| posiciones tranquilas | 0.693 |
| blancas al turno | 0.510 |

Solo se admiten posiciones legales de ajedrez estándar (`chess.Board(fen).is_valid()`); lo
descartado se cuenta en los metadatos del shard (`dropped`, `illegal`).

**Comandos.**

```bash
# 1. descarga reanudable (4 conexiones: con más, Lichess responde 429) y conversión a shards
aria2c -x 4 -s 4 -k 64M --continue=true --auto-file-renaming=false --file-allocation=none \
    --max-tries=0 --retry-wait=30 -d data/lichess -o lichess_db_eval.jsonl.zst \
    https://database.lichess.org/lichess_db_eval.jsonl.zst
python scripts/data/build_raw_shards.py --input data/lichess/lichess_db_eval.jsonl.zst \
    --output-dir data/lichess/raw --log-file data/lichess/progress.log

# 2. sidecars (los shards que ya tienen sidecar se saltan)
python scripts/data/build_sidecars.py --raw data/lichess/raw --out data/lichess/sidecar --jobs 4
```

Opciones útiles de `build_raw_shards.py`: `--rows` (solo las primeras N líneas), `--shard-rows`
(por defecto 1,000,000), `--workers`, `--url` (en lugar de `--input`, lee por HTTP, más lento y
frágil). Cada shard se escribe en un `.tmp` y se renombra, así que existe completo o no existe;
repetir el comando salta los shards terminados. El `.zst` descargado (22.4 GB) es temporal: se
borra cuando los shards están comprobados. `train` lee el flujo de filas TRAIN de los sidecars y
los shards, y `evaluate` juzga sobre una muestra con semilla de 200,000 filas de test (semilla 42).

## Escala de esfuerzo

`--effort N` entrena con **N millones de posiciones** (mínimo 1 M) y `1 + N // 10` épocas. Esfuerzo
15 = 15 M posiciones, 2 épocas; esfuerzo 30 = 30 M, 4 épocas. Es la escala de todas las tablas de
esta página salvo que se indique otra.

```bash
python -m chess_engine.training.train --model linear_pst_terms --effort 15
```

## Juez estático

200,000 posiciones de test, esfuerzo 15, semilla 42 salvo indicación. Spearman entre la predicción
y la evaluación de Stockfish (todas las posiciones):

| modelo | Spearman | notas |
|---|---|---|
| `linear_pst_terms` | 0.655 | semilla 7: 0.645 |
| `linear_pst_terms@mates10` | 0.646 | semilla 7: 0.650 |
| `linear_pst_terms@quiet` | 0.651 | entrenado solo con posiciones tranquilas |
| `net4_pst` | 0.627 | |
| `linear_pst` | 0.623 | |
| `linear_king_bucket` | 0.609 | |
| `h1` (heurística de referencia) | 0.561 | |
| `linear_material` | 0.475 | esfuerzo 15 |
| `linear_material@e90` | 0.474 | esfuerzo 90: MAE 0.158, mejor jugada 0.191 (0.549 no tranquilas); no mejora a esfuerzo 15 |
| `linear_pst_terms@e30` | — | entrenado a esfuerzo 30; sin juicio estático ni partidas |

Acierto de mejor jugada y MAE (sobre la escala de puntuación esperada) están en las fichas de cada
modelo, sección `accuracy` (`src/chess_engine/models/cards/<modelo>.json`).

```bash
python -m chess_engine.training.train --model linear_pst_terms --effort 15
python -m chess_engine.training.train --model linear_pst_terms --effort 15 --seed 7 --tag s7
python -m chess_engine.training.train --model linear_pst_terms --effort 15 --representation '{"filter": "quiet"}' --tag quiet
python -m chess_engine.comparison.evaluate --model linear_pst_terms
python -m chess_engine.comparison.evaluate --model linear_pst_terms@s7
```

El resto de modelos se entrenan y evalúan igual cambiando `--model` (`linear_material`,
`linear_pst`, `linear_king_bucket`, `net4_pst`; `h1` solo se evalúa).

## Mates al 10 %

`linear_pst_terms@mates10` conserva el 10 % de las posiciones de mate en TRAIN y validación:

```bash
python -m chess_engine.training.train --model linear_pst_terms --effort 15 --mate-keep 0.1 --tag mates10
python -m chess_engine.training.train --model linear_pst_terms --effort 15 --mate-keep 0.1 --seed 7 --tag mates10s7
```

Con las dos semillas el acierto de mejor jugada sube (+1 a +2 puntos en total, +3 a +6 en
posiciones no tranquilas) y el MAE empeora ligeramente (0.133 y 0.132 frente a 0.129 y 0.131).

**Validación en partidas.** Esfuerzo 15, 100 partidas, 50 aperturas (dos partidas por apertura,
colores cambiados), ancla `linear_pst_terms`, 60 s + 1 s por jugada. Registro local:
`.superpowers/night1/MATES_PLAY.md` (no versionado).

| jugador | puntuación % | V-T-D | intervalo 95 % | Elo frente a `linear_pst_terms` |
|---|---|---|---|---|
| `linear_pst_terms@mates10` | 71.0 | 64-14-22 | 62.9 - 79.1 | +156 |
| `linear_pst_terms` (100 % de los mates, original) | 29.0 | 22-14-64 | 20.9 - 37.1 | 0 |

Ninguna partida acabó por forfeit, tiempo o jugada ilegal (0 de 100). Terminaciones: 86 jaque
mate, 12 triple repetición, 1 cincuenta jugadas, 1 material insuficiente.

```bash
python scripts/bench/play_tournament.py --players linear_pst_terms,linear_pst_terms@mates10 \
    --anchor linear_pst_terms --openings 50 --workers 5 --base 60 --increment 1
```

**Decisión.** El entrenamiento por defecto usa ahora el 10 % de las posiciones de mate
(`DEFAULT_MATE_KEEP = 0.10`), así que este comando produce el modelo del 10 %:

```bash
python -m chess_engine.training.train --model linear_pst_terms --effort 15
```

El `linear_pst_terms` original (100 % de los mates) se retira del repositorio; se recupera del
historial de git en el commit `03169db` (`linear_pst_terms.npz` y `linear_pst_terms.json`). Una
variante al 100 % se entrena con `--mate-keep 1.0 --tag mates100`.

**Alcance.** Es un solo par de jugadores, una sola escala (esfuerzo 15) y 100 partidas. Este
test no cambia la clasificación de los demás modelos.

## Esfuerzo 90 (`linear_material`)

Entrenado con 90 M de posiciones y 10 épocas. Con el mismo juez que la tabla anterior: Spearman
0.474, MAE 0.158, acierto de mejor jugada 0.191 en total y 0.549 en posiciones no tranquilas. La
pérdida de validación queda plana (0.6722 a 0.6719), así que el esfuerzo extra no añade información
para este modelo: se queda con el esfuerzo 15.

```bash
python -m chess_engine.training.train --model linear_material --effort 90
python -m chess_engine.comparison.evaluate --model linear_material@e90
```

**Reproducibilidad.** Las fichas `linear_material@e90` y `linear_pst_terms@e30` se generaron con el commit
`1dd8cb8` y cambios sin confirmar (`+dirty`). Los refactors posteriores (`cba6d3a`, `d56a140`,
`428bffb`) cambian el código pero no los números declarados; no se han vuelto a medir.

## Quiescencia a profundidad fija

Acuerdo de mejor jugada con Stockfish en 2,000 posiciones de test, comparación pareada (cada
posición se juega con y sin quiescencia). Registros locales: `.superpowers/night1/players2000.log` y
`.superpowers/night1/quiet_search_grid.log` (no versionados).

| bot | profundidad 3, quiescencia 4 frente a 0 | error estándar pareado |
|---|---|---|
| `h1` | +0.060 | 0.010 |
| `linear_pst` | +0.060 | 0.009 |
| `linear_pst_terms` | +0.062 | 0.009 |

En `linear_pst_terms` (rejilla): profundidad 2 con quiescencia 4 supera a profundidad 3 sin ella
(+0.037, SE 0.013) con la mitad de tiempo (0.081 s frente a 0.188 s por jugada). Detalle del
algoritmo en [SEARCH_ALGORITHMS.md](SEARCH_ALGORITHMS.md).

## Torneo

Todos contra todos a igual tiempo por jugada (60 s + 1 s), aperturas emparejadas (dos partidas por
apertura, colores cambiados), sin partidas perdidas por forfeit. Puntuación = victorias + medias tablas.

**Ronda 1:** `h1` y cinco modelos, 20 partidas por pareja (300 partidas, 0 forfeits).

| jugador | puntuación % |
|---|---|
| `linear_material` | 69.0 |
| `linear_king_bucket` | 59.0 |
| `net4_pst` | 49.0 |
| `linear_pst` | 48.5 |
| `linear_pst_terms` | 37.5 |
| `h1` | 37.0 |

**Ronda 2:** los tres primeros, 50 partidas por pareja (150 partidas, 0 forfeits).

| jugador | puntuación % |
|---|---|
| `linear_material` | 56.5 |
| `net4_pst` | 51.0 |
| `linear_king_bucket` | 42.5 |

```bash
python scripts/bench/play_tournament.py --players h1,linear_material,linear_pst,linear_pst_terms,linear_king_bucket,net4_pst \
    --anchor h1 --base 60 --increment 1 --workers 5 --openings 10 \
    --out data/tournament/games_20.jsonl --report TOURNAMENT_20.md --logs data/tournament/games_20
python scripts/bench/play_tournament.py --players linear_material,net4_pst,linear_king_bucket \
    --anchor linear_material --base 60 --increment 1 --workers 5 --openings 25 \
    --out data/tournament/games_50.jsonl --report TOURNAMENT_50.md --logs data/tournament/games_50
```

(`--openings 10` son 20 partidas por pareja y `--openings 25` son 50.)

## Discrepancia: velocidad frente a evaluación

El juez estático pone a `linear_pst_terms` el primero y a `linear_material` el último; jugando a tiempo
fijo es al revés. La explicación que los datos sostienen: `linear_material` evalúa unos 16 k nodos por
segundo frente a 9 k de `linear_pst_terms` (1 s por jugada, 200 posiciones de test), así que en el mismo
tiempo alcanza unos 0.6 semiplies más. Ese extra de profundidad es lo que le da la ventaja en el torneo.

Prueba con profundidad fija (3 para los dos, sin reloj efectivo, 50 aperturas, 100 partidas por pareja):

| jugador | puntuación % a profundidad 3 | puntuación % a 60 s + 1 s |
|---|---|---|
| `linear_pst_terms` | 74.8 | 37.5 (ronda 1) |
| `linear_pst` | 38.8 | 48.5 (ronda 1) |
| `linear_material` | 36.5 | 69.0 (ronda 1) |

A igual profundidad, `linear_material` pierde con claridad (14-20-66 frente a `linear_pst_terms`). La
ventaja de `linear_material` a tiempo fijo es, por tanto, de velocidad y no de evaluación. Lo medido
solo cubre profundidad 3 y 1 s por jugada; no se ha medido el cruce con profundidades mayores.

```bash
python scripts/bench/play_tournament.py --players linear_material,linear_pst_terms,linear_pst \
    --anchor linear_pst_terms --max-depth 3 --base 600 --increment 0 --openings 50 --workers 5 \
    --out data/tournament/depth3.jsonl --report data/reports/DEPTH3.md
```

El torneo a profundidad fija no actualiza las fichas de modelo. Medición de velocidad:
`.superpowers/experimentos/velocidad/` (no versionada).

## Tabla de modelos

Generada por `python -m chess_engine.comparison.report` a partir de las fichas; no se edita a mano.

<!-- BEGIN GENERATED TABLES -->
| Model | Kind | Params | Representation | Effort | Spearman | MAE | Best move, all / non-quiet | µs per eval | Depth 1 s | Depth 3 s | Depth 10 s | Search agreement, 10 s | Score | Elo ± interval |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear_king_bucket` | plays | 10,369 | winprob-k0.00368-stm-all | 15 | 0.609 | 0.143 | 17.9% / 47.1% | 35.2 | — | — | — | — | 59.0% | +134 ± 67 |
| `linear_material` | plays | 6 | winprob-k0.00368-stm-all | 15 | 0.475 | 0.156 | 19.1% / 54.9% | 21.0 | — | — | — | — | 69.0% | +198 ± 72 |
| `linear_material@e90` | plays | 6 | winprob-k0.00368-stm-all-mates10 | 90 | 0.474 | 0.158 | 19.1% / 54.9% | 22.1 | — | — | — | — | — | — |
| `linear_pst` | plays | 769 | winprob-k0.00368-stm-all | 15 | 0.623 | 0.137 | 18.8% / 50.6% | 57.2 | — | — | — | — | 48.5% | +70 ± 66 |
| `linear_pst_terms` | plays | 791 | winprob-k0.00368-stm-all-mates10 | 15 | 0.646 | 0.133 | 20.0% / 54.4% | 114.7 | — | — | — | — | 71.0% | +156 ± 70 |
| `linear_pst_terms@e30` | plays | 791 | winprob-k0.00368-stm-all-mates10 | 30 | — | — | — | — | — | — | — | — | — | — |
| `linear_pst_terms@mates10s7` | plays | 791 | winprob-k0.00368-stm-all-mates10 | 15 | 0.650 | 0.132 | 20.2% / 54.8% | 118.5 | — | — | — | — | — | — |
| `linear_pst_terms@quiet` | plays | 791 | winprob-k0.00368-stm-quiet | 15 | 0.651 | 0.128 | 19.8% / 53.5% | 117.1 | — | — | — | — | — | — |
| `linear_pst_terms@s7` | plays | 791 | winprob-k0.00368-stm-all | 15 | 0.645 | 0.131 | 18.4% / 49.1% | 115.2 | — | — | — | — | — | — |
| `net4_pst` | plays | 206,177 | winprob-k0.00368-stm-all | 15 | 0.627 | 0.133 | 18.9% / 48.1% | 68.4 | — | — | — | — | 49.0% | +73 ± 64 |
| `cnn_resnet` | accuracy record | 572,353 | — | — | 0.371 | 0.165 | 20.3% / 40.4% | 24,407.1 | — | — | — | — | — | — |
| `mlp_features` | accuracy record | 1,505 | — | — | 0.614 | 0.130 | 17.9% / 43.6% | 124.2 | — | — | — | — | — | — |
<!-- END GENERATED TABLES -->
