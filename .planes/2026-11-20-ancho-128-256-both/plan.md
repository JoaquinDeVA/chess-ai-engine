# Ancho 128 y 256 con both

## Objetivo
Saber si una red más ancha (128 o 256 unidades en el acumulador, dos perspectivas, entradas `pst_slotted`) juega mejor que `net_pst_w64_both`, entrenando y midiendo todo en el nodo. Hecho cuando: `python -m chess_lab.measure.report` muestra las dos redes con su val loss, Spearman y tiempo a profundidad, y hay un SPRT [0, 5] de cada red que pase el cribado contra `net_pst_w64_both@e30`, con las cifras en `docs/RESULTS.md` y la guía.

## Fuera de alcance
- Amenazas (`pst_threats`) con el ancho nuevo: solo si gana una de estas dos; plan aparte.
- Cubetas de rey, tercera capa, SCReLU y el resto de ablaciones de «Decisiones abiertas».
- Cambios de datos y de filtros; `effort 100` (con 64 unidades no mejoró sobre `e30`).
- Cambiar la red que juega en el bot o la incrustada en el binario.

## Decisiones
- Receta `--effort 30 --lr-decay 0.5`, la de la base `net_pst_w64_both@e30` (val 0,63102) — descartado: effort 100 (4,6 h por red, sin ganancia a 64; se prueba solo si el ancho nuevo sigue bajando la pérdida).
- Línea base de los SPRT: `net_pst_w64_both@e30`, mismo binario, reloj 10 s + 0,2 s, Hash 16, un hilo, SPRT [0, 5] — descartado: compararlas con `threats` (mezcla ancho y entradas).
- Cribado barato antes del SPRT: pasa a SPRT la red cuya val loss baje al menos 0,0005 de 0,63102 y cuyo tiempo a profundidad 11 (`uci ttd 11`) no suba más de un 50 % (umbrales míos, sin medir antes; ajustables).
- La clave de la caché de lotes depende del conjunto de rasgos y no de la clase: `training/cache.py` usa hoy `model_class.__name__`, así que `w128` volvería a codificar las mismas 30 M de posiciones que `w64` en otra carpeta. La codificación solo depende de `feature_set`, la representación, el número de filas, la semilla, el filtro de tranquilas, `mate_keep` y el lote; no del ancho ni de `perspective` — descartado: añadir el ancho a la clave (duplicaría cada caché).
- Las cachés ya creadas en el nodo se renombran al hash nuevo (se calcula con la misma función) en vez de recodificar; si el nodo no las tiene (a mirar en la fase 1), no hay nada que migrar.
- Todo el entrenamiento y las medidas en el nodo (`pueue`, `.computo/scripts/`); en local solo código, tests y docs.
- El round robin y el bot comparten CPU con el codificado: no se lanza nada de ello a la vez que el entrenamiento (en el nodo el bot y el round robin constan parados al planificar).

## Fases
### 1. Clave de caché por rasgos
Cambiar `cache_path` para que use `model_class.feature_set` en lugar del nombre de clase; test nuevo en `tests/lab/training/test_cache.py`: dos clases con el mismo `feature_set` y distinto `hidden` dan la misma ruta; `pst_slotted` y `pst_threats` dan rutas distintas; el resto de entradas siguen cambiando la ruta. Mirar en el nodo (solo lectura, `du` y `ls` de `~/chessai/train/repo/data/cache`) qué cachés hay y, si hay, añadir a `.computo/scripts/` un script que las renombre al hash nuevo.
Ficheros: `src/chess_lab/training/cache.py`, `tests/lab/training/test_cache.py`, `.computo/scripts/` · Check: `pytest tests/lab/training/test_cache.py -q` → todo pasa; el script nuevo, en seco, lista los renombres sin ejecutarlos.

### 2. Definir w128 y w256
Dos clases en `src/chess_lab/models/definitions.py` (`net_pst_w128_both`, `net_pst_w256_both`; `pst_slotted`, `hidden` 128 y 256, `perspective = True`), registradas en `MODELS` y con su descripción de arquitectura. Rust no cambia: el ancho sale del JSON y el producto escalar va de 8 en 8 (comprobado en `rust/src/eval/net.rs`).
Ficheros: `src/chess_lab/models/definitions.py`, `tests/lab/models/test_definitions.py` · Check: `pytest tests/lab/models tests/lab/training -q` → todo pasa, incluida una prueba de forma de capas (768 → 128/256 → 1 con 256/512 de entrada a la salida).

### 3. entrenar.sh con effort y modelos elegibles
`.computo/scripts/entrenar.sh` fija hoy `effort 100` y una sola red: leer `EFFORT` (por defecto 100, como ahora) y que `SOLO` elija las redes ya existe. Actualizar la cabecera con la receta de esta serie.
Ficheros: `.computo/scripts/entrenar.sh` · Check: `bash -n .computo/scripts/entrenar.sh` → sin errores; ejecución en seco en el nodo (`entrenar.sh prueba` con `SOLO=net_pst_w128_both`) → ficha, `.npz` y JSON UCI creados y cargados por `uci`.

### 4. Entrenar en el nodo
Subir lo necesario según `.computo/README.md`, lanzar con `pueue` `EFFORT=30 SOLO=net_pst_w128_both,net_pst_w256_both` (una tras otra, GPU), anotar cada acción en `.computo/registro.md`. Tiempo estimado: de 3 a 6 h por red (la de 64 tardó 1,5 h a e30; estimado, el ancho cuesta más por época). Si `w256` no cabe o tarda demasiado, parar con SIGINT y decidirlo aquí.
Ficheros: `data/results/entrenar/<etiqueta>/` (traído con `reunir`) · Check: fichas `net_pst_w128_both@e30.json` y `net_pst_w256_both@e30.json` con `val_losses` completas; la caché del lote usada es la de `w64` (sin carpeta nueva de caché en el nodo).

### 5. Cribado: pérdida y tiempo a profundidad
Medir las dos redes y la base con `python -m chess_lab.measure.evaluate` (Spearman, MAE, mejor jugada) y `uci ttd 11` / `uci bench` con el mismo binario y la misma máquina, alternando ejecuciones. Aplicar el umbral de «Decisiones».
Ficheros: fichas en `src/chess_lab/models/store/cards/`, `data/results/` · Check: tabla de tres filas con val loss, Spearman, nodos/s y tiempo a profundidad 11, mediana de al menos 3 ejecuciones; la decisión de cada ancho (pasa / no pasa) por escrito.

### 6. SPRT de las redes que pasen
Un trabajo por red en `queue.json` del round robin (`sp_w128_vs_w64`, `sp_w256_vs_w64`), [0, 5], 10 + 0,2, Hash 16, un hilo, mismo binario y suite de aperturas que `sp_*` anteriores; instalar y lanzar con `instalar.sh` y `arrancar.sh`, sin el relevo del bot (`ARRANCAR_RR` vacío) hasta terminar. Parar cuando el LLR cruce ±2,94 o tras un máximo de rondas fijado en la cola; el valor, si no cruza, se cuenta como «válido» solo con la regla de `docs/MEASUREMENT.md`.
Ficheros: `src/chess_lab/measure/queue.json`, `data/results/roundrobin/` (traído con `reunir`) · Check: `bash .computo/scripts/estado.sh` → cada SPRT con su veredicto; `llr.csv` traído.

### 7. Documentar
Cifras a `docs/RESULTS.md` (la tabla sale de `python -m chess_lab.measure.report`) y `CHANGELOG.md` si hay red nueva visible; en la guía, pestañas Modelos (§8–9), Entrenamiento (§7, la clave de la caché por rasgos) y «Decisiones abiertas» (mover lo medido; ancho 128 ya tiene cifra). Leer `guia/NORMAS.md` antes de editar.
Ficheros: `docs/RESULTS.md`, `CHANGELOG.md`, `guia/paginas/{modelos,entrenamiento,abiertas}.html` · Check: `pytest tests/test_docs.py -q`, `python guia/check_docs.py --guia` y la guía en el navegador sin errores de consola.

### 8. Cierre
Suite completa una vez: `pytest -m "not slow"`, `ruff check . && ruff format --check .`. Después `andamio:revisor` con la ruta de este plan y el rango de commits.

## Estado
Fase actual: 2 (fase 1 hecha: clave por `feature_set` y test; el nodo no tiene `data/cache` en `train/repo`, no hay nada que migrar y no se creó script). Trampas encontradas: la caché de hoy no se comparte entre anchos (clave con el nombre de la clase); `entrenar.sh` fija effort 100; en el nodo el bot y el round robin constan parados.
