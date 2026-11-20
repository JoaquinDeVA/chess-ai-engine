# Entradas con acumulador: material, peones por columna y cubetas de rey propio

## Objetivo
Saber cuánto aporta cada familia de entradas nueva (material/fase `M`, peones por columna `P`, cubetas del rey propio `K`) y cómo interactúan, sobre `pst_slotted` con `both`, usando un único `feature_set` (`pst_ctx`) y una sola caché. Hecho cuando: `python -m chess_lab.measure.report` muestra los modelos de la etapa 1 (y de la 2 si procede) con val loss, Spearman y tiempo a profundidad, y hay SPRT [0, 5] de los que pasen el cribado contra `net_pst_w128_both@e15`, con las cifras en `docs/RESULTS.md` y la guía.

## Fuera de alcance
- Amenazas combinadas con las familias nuevas; peón pasado; movilidad; tercera capa; SCReLU.
- Reentrenar los modelos `pst_slotted` (w64, w128, w256): `pst_ctx` sin familias es el mismo modelo.
- Cambiar la red que juega en el bot o la incrustada en el binario.
- Todo lo del plan `2026-11-20-ancho-128-256-both` (sigue su curso: w256 pst entrenando; su cribado y su SPRT de w128 pst).

## Decisiones
- Un solo `feature_set` `pst_ctx` que codifica todo; cada modelo declara las familias activas (`families`) y la capa las enmascara. La clave de la caché ya depende solo del `feature_set` (commit 7eb4a8e): una caché para todos — descartado: un `feature_set` por combinación (7 cachés de ~14 GB y 7 codificados en CPU).
- Diseño de entradas, todo disperso (valor 1.0), sin entradas densas, así la vista del rival sale intercambiando bloques como hoy:
  - Piezas: los 768 de siempre.
  - `M`: por bando, una entrada por tipo y número de piezas (cuenta 0 se omite): peón 1–8, caballo/alfil/torre 1–2 (2 = «2 o más»), dama 1 (= «1 o más»): 15 por bando, 30 en total. La fase es función del material.
  - `P`: por bando y columna, una entrada si hay 1 peón y otra si hay 2 o más: 16 por bando, 32 en total.
  - `K`: 4 cubetas por el rey **propio** de cada vista, según su casilla orientada: enroque corto (f1–h1), largo (a1–c1), casa (d1, e1), otra. Se guardan como 2 entradas de índice (cubeta del que mueve y del rival) y la capa las lee: la fila de una pieza es `cubeta × 768 + pieza`. Con `K` activa la tabla tiene `4 × 768` filas de piezas — la cubeta por el rey **rival** (k4) ya perdió −31,9 ± 21,5 frente a w64; las 4 cubetas y su reparto son míos, ajustables.
  - Máximo de entradas por posición ≈ 32 + 10 + 16 + 2 = 60 (frente a 32): la caché crece a ~26 GB estimados (el nodo tiene 284 GB libres).
- Rust: `FeatureSet::PstCtx` con las familias leídas del JSON (`families`); `M` y `P` se actualizan por jugada (se quita la entrada vieja y se pone la nueva de lo que cambia); `K` rehace el acumulador de una vista solo cuando cambia su cubeta (no en cada jugada de rey).
- Línea base de todo: `net_pst_w128_both@e15` (mismos datos, receta y ancho, ya entrenada); para w256, `net_pst_w256_both@e15` cuando acabe — descartado: comparar con `w64@e30` (otros datos).
- Etapa 1, ancho 128, 4 modelos: `+M`, `+P`, `+K`, `+MPK` (nombres `net_ctx_{m,p,k,mpk}_w128_both`). Si `+MPK` ≈ lo mejor o ≈ la suma de los individuales no se entrenan los pares; si no, `+MP`, `+MK`, `+PK` (3 más, decidido con las cifras) — descartado: factorial completo de 14 modelos (unas 36–42 h solo de entrenar, estimado).
- Etapa 2, ancho 256: solo las 2 o 3 mejores combinaciones de la etapa 1, tras el cribado.
- Receta como la serie de ancho: `--effort 15 --epochs 40 --lr-decay 0.5` sobre el dataset nuevo. Cribado: pasa a SPRT quien baje ≥ 0,0005 la val loss de la base y no suba más de un 50 % el tiempo a profundidad 11 (umbrales míos, sin medir; los de la serie de ancho).
- Todo el entrenamiento y los SPRT en el nodo; en local, código, tests y docs. Round robin y bot parados mientras se codifica o entrena.
- «Alinear origen y nodo» = que el nodo (destino) tenga el mismo código que el repo local (origen), con `rsync` según `.computo/README.md`. Sin `git push`: no se toca `origin` de GitHub. No se sube `train/repo` mientras w256 entrena (`entrenar.sh` lo sincroniza y recompila al arrancar): se sube `~/chessai/repo` ahora y `train/repo` en el siguiente `entrenar.sh`.

## Fases
### 1. `pst_ctx` en Python
`models/features.py`: clase `CtxPstFeatures` (nombre `pst_ctx`, 768 + 30 + 32 + 8 entradas, `max_entries` 60) con el diseño de «Decisiones»; registrada en `FEATURE_SETS`. Las piezas, en el mismo orden que `pst_slotted`.
Ficheros: `src/chess_lab/models/features.py`, `tests/lab/models/test_features.py` · Check: `pytest tests/lab/models/test_features.py -q` → pasa, con casos a mano (posición inicial: material, columnas y cubetas esperadas; tras un enroque, tras una promoción y tras una captura de peón).

### 2. Modelos y capa de entrenamiento
`models/definitions.py`: atributo `families` en las clases y 4 clases de la etapa 1 (`NnCtx{M,P,K,MPK}W128BothModel`, `feature_set = "pst_ctx"`, `perspective = True`), en `MODELS`; forma de la primera capa según `families`. `training/neural.py`: la capa del acumulador enmascara las familias no activas, la vista del rival intercambia los bloques de `M` y `P` y las dos cubetas, y con `K` direcciona `cubeta × 768 + pieza`. `models/export_uci.py`: escribe `families` en el JSON.
Ficheros: `src/chess_lab/models/{definitions,export_uci}.py`, `src/chess_lab/training/neural.py`, `tests/lab/models/test_definitions.py`, `tests/lab/training/test_neural.py` · Check: `pytest tests/lab/models tests/lab/training -q` → pasa, con: (a) `pst_ctx` sin familias da la misma salida que `pst_slotted` con los mismos pesos; (b) intercambiar bandos de una posición intercambia las dos vistas; (c) formas y número de parámetros.

### 3. Rust
`rust/src/eval/features.rs`: `FeatureSet::PstCtx` con familias; `net.rs`: lee `families` del JSON y comprueba `fan_in`; `accumulator.rs`: actualización por jugada de `M` y `P`, y rehacer la vista cuando cambia la cubeta de `K`; la ruta de `NetState::logit` con `layers.len() == 2` no cambia. `python.rs` si expone los rasgos.
Ficheros: `rust/src/eval/{features,net,accumulator}.rs`, `rust/src/python.rs`, `tests/core/` · Check: en `rust/`: `cargo test --release`, `cargo clippy --all-targets --features python -- -D warnings`, `cargo fmt --check`; `pytest tests/core -q`. Pruebas nuevas: los rasgos de Rust igualan a los de Python en un lote de FEN (incluye enroque, promoción, captura al paso), y la puntuación incremental tras una partida de jugadas aleatorias iguala a `Net::score` (como la prueba que ya hay) para cada combinación de familias.

### 4. Alinear local y nodo
`ruff check . && ruff format --check .`, commit local de lo anterior (sin push), subir `~/chessai/repo` al nodo (`src/`, `rust/{src,models,Cargo.toml,Cargo.lock}`, `.computo/scripts/`, `pyproject.toml`) y anotarlo en `.computo/registro.md`. Compilar el binario del nodo en `~/chessai/repo` y comprobar `uci` con una red `pst_ctx` de prueba.
Ficheros: `.computo/registro.md` · Check: `git status` limpio; `rsync -rcn --itemize-changes` de esos directorios hacia el nodo no lista diferencias; en el nodo `uci` carga el JSON de prueba y responde `bench`.

### 5. Colas de entrenamiento (etapa 1)
Ampliar `entrenar.sh` si hace falta (lista de modelos de la serie). Prueba en seco en el nodo (`entrenar.sh prueba` con `SOLO=net_ctx_mpk_w128_both`, cuando w256 haya acabado o con tu OK para esperar a ello), y después `pueue add --after <w256>` con `EFFORT=15 EPOCHS=40 SOLO=net_ctx_m_w128_both,net_ctx_p_w128_both,net_ctx_k_w128_both,net_ctx_mpk_w128_both`: el primero crea la caché `pst_ctx` (codificado en CPU), los demás la reutilizan. Cada acción, una línea en el registro.
Ficheros: `.computo/scripts/entrenar.sh`, `.computo/registro.md` · Check: `pueue status` → la tarea en cola; la prueba en seco acaba con ficha, `.npz` y JSON cargados por `uci`; tras la primera época de `m`, una sola carpeta nueva en `data/cache`.

### 6. Entrenar y cribar la etapa 1
Esperar a las 4 redes, traerlas con `reunir`/`rsync` de `data/results/entrenar/e15/`, corregir el `commit` de las fichas, y cribar contra `net_pst_w128_both@e15` con `python -m chess_lab.measure.evaluate` y `uci ttd 11` (mediana de ≥ 3 ejecuciones, mismo binario y máquina). Decidir por escrito: pares sí/no (según «Decisiones») y qué 2 o 3 combinaciones pasan a w256.
Ficheros: fichas en `src/chess_lab/models/store/cards/` · Check: tabla de las 5 filas (base + 4) con val loss, Spearman, tiempo a profundidad 11 y la decisión.

### 7. Etapa 2 (pares si procede, y w256)
Definir las clases que falten, repetir fases 4–5 para ellas (subida, prueba en seco, cola) y 6 (traer y cribar). Los w256 se comparan con `net_pst_w256_both@e15`.
Ficheros: `definitions.py`, `entrenar.sh` · Check: tabla igual que la de la fase 6.

### 8. SPRT de los que pasen
Un trabajo por red en `queue.json` del round robin, [0, 5], 10 s + 0,2 s, Hash 16, un hilo, mismo binario, contra su base del mismo ancho; sin relevo del bot (`ARRANCAR_RR` vacío). Parar con LLR ±2,94 o con la regla «válido» de `docs/MEASUREMENT.md`.
Ficheros: `src/chess_lab/measure/queue.json`, `data/results/roundrobin/` · Check: `bash .computo/scripts/estado.sh` → cada SPRT con veredicto; `llr.csv` traído.

### 9. Documentar y cerrar
Cifras a `docs/RESULTS.md` (tabla de `measure.report`) y `CHANGELOG.md` si hay red visible; guía: pestañas modelos, entrenamiento (el `feature_set` único y el enmascarado) y «Decisiones abiertas»; leer `guia/NORMAS.md` antes. Después `pytest -m "not slow"`, `ruff check . && ruff format --check .` y `andamio:revisor` con la ruta del plan y el rango de commits.
Ficheros: `docs/RESULTS.md`, `CHANGELOG.md`, `guia/paginas/{modelos,entrenamiento,abiertas}.html` · Check: `pytest tests/test_docs.py -q`, `python guia/check_docs.py --guia`, guía sin errores de consola.

## Tiempos estimados
Con 2,5 h por modelo (tu cifra; w128 pst midió 1,45 h, pero `pst_ctx` lleva casi el doble de entradas por fila):

| Bloque | Estimado |
|---|---|
| Fases 1–5 (código, Rust incremental, subida) | trabajo mío, sin nodo; la Rust es lo largo |
| Codificado de la caché `pst_ctx` (primera época) | 1–2 h (CPU) |
| Etapa 1: 4 modelos w128 | 10 h |
| Pares (si procede): 3 modelos | 7,5 h |
| Etapa 2: 2–3 modelos w256 | 8–11 h (w256 más lento por época, sin medir) |
| SPRT: 3–4 a 3–6 h cada uno | 9–24 h (estimado; sin medir con estos modelos) |
| **Total de nodo, camino sin pares** | **≈ 30–48 h** |
| Total con factorial completo (14 modelos), solo entrenar | ≈ 36–42 h, más SPRT |

## Estado
Fase actual: 1 (sin empezar). Trampas conocidas: `entrenar.sh` sincroniza y recompila `train/repo` al arrancar, no hacerlo con w256 entrenando; el JSON de un modelo `pst_ctx` sin `families` debe tratarse como sin familias; la caché crece a ~26 GB (estimado).
