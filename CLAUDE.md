# ChessAIEngine

Motor de ajedrez con búsqueda adversarial, evaluaciones clásicas y aprendidas, y pipeline de datos de la base de evaluaciones de Lichess. Estado: activo.

## Antes de tocar nada
1. `git pull` (remoto `origin`, rama `main`).
2. Si existe `vault/` en tu copia local (no está versionado), lee `vault/README.md` y la ficha del módulo que vayas a tocar. Solo si no basta, lee el código.
3. Si el vault y el código discrepan, manda el código; avísame para actualizar la ficha.

## Cómo trabajar
- Tarea no trivial: `superpowers:brainstorming`, luego `superpowers:writing-plans`; espera mi OK antes de ejecutar.
- Cambios quirúrgicos: toca solo lo que pide la tarea y respeta el estilo existente.
- Procesos largos (entrenamiento, `scripts/run_all.sh`, torneos): uno pesado a la vez. Lánzalos con `nohup setsid ... > log 2>&1 &` y mira el log. No uses `pgrep -f` con un patrón que aparezca en tu propio comando: espera para siempre.

## Subagentes
- Modelo por defecto `haiku`. `sonnet` solo si la tarea exige juicio (diseño, decidir entre opciones). Nunca dejes el modelo sin indicar.
- Brief con rutas exactas, comandos exactos y criterio de aceptación. Un encargo, un objetivo.
- Nunca dos subagentes escribiendo los mismos ficheros a la vez.
- No des nada por hecho con su palabra: ejecuta tú el test, el lint o el grep y mira la salida antes de decir «hecho».
- No digas «sincronizado» ni «actualizado» sin haber corrido el comando que lo demuestra.
- Si la tarea cabe en una respuesta, hazla tú: no lances subagentes por inercia.

## Git
- Rama única `main`. Un cambio por commit, con los ficheros por ruta (nunca `git add -A`).
- Mensajes en inglés, estilo `tipo: descripción`, como el historial.
- Sin líneas `Co-Authored-By` ni "Generated with". Push solo si lo pido.

## No leer (salvo necesidad concreta)
- `data/`: unos 24 GB de datos crudos y sidecars. No borrar nada ahí.
- `src/chess_engine/models/artifacts/*.keras`: binarios de modelos.
- `data/opening_book.json`: 2.339 líneas de libro de aperturas; consulta por FEN, no lo leas entero.
- `docs/img/` y `docs/benchmarks/`: salidas generadas.

## Comprobar
- Instalar: `pip install -e ".[dev]"`.
- Durante una tarea: solo los tests de los ficheros que tocas o que los consumen (ver tabla). Nunca la suite completa.
- Al terminar un plan: el coordinador lanza la suite completa una vez (`pytest -m "not slow"`), no los subagentes.
- Lint: `ruff check . && ruff format --check .`
- Ejecutar: `python main.py`.

**Qué es un test aquí:** comprueba funcionalidad del código (una función, un formato, un rechazo de argumentos). No entrena modelos, no ejecuta CLIs completas, no juzga si un modelo o un flujo es bueno (eso es la comparación, no un test). Si necesitas ejecutar algo pesado, usa un mock o un doble del modelo. Un test que tarda más de unos segundos está mal diseñado.

| Si tocas | Tests |
|---|---|
| `evaluation/`, `bots/`, `comparison/tradeoff.py` | `tests/test_heuristics_tuned.py`, `tests/test_bots.py`, `tests/test_tradeoff.py` |
| `comparison/tournament.py` | `tests/test_tournament.py` |
| `models/` | `tests/test_models.py`, `tests/test_linear_eval.py`, `tests/test_neural_eval.py` |
| `training/` | `tests/test_training.py`, `tests/test_sidecar.py`, `tests/test_effort.py`, `tests/test_lichess_raw.py` |
| `search/` | `tests/test_advanced_search.py`, `tests/test_chess_state_key.py`, `tests/test_terminal_detection.py` |
| `docs/`, `README.md` | `tests/test_docs.py` |
