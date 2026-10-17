# ChessAIEngine

Motor de ajedrez con búsqueda adversarial, evaluaciones clásicas y aprendidas, y pipeline de datos de la base de evaluaciones de Lichess. Estado: activo.

## Antes de tocar nada
1. `git pull` (remoto `origin`, rama `main`).
2. Lee `vault/README.md` y la ficha del módulo que vayas a tocar. Solo si no basta, lee el código.
3. Si el vault y el código discrepan, manda el código; avísame para actualizar la ficha.

## Cómo trabajar
- Tarea no trivial: `superpowers:brainstorming`, luego `superpowers:writing-plans`; espera mi OK antes de ejecutar.
- Cambios quirúrgicos: toca solo lo que pide la tarea y respeta el estilo existente.
- Procesos largos (entrenamiento, `scripts/run_all.sh`, torneos): uno pesado a la vez. Lánzalos con `nohup setsid ... > log 2>&1 &` y mira el log. No uses `pgrep -f` con un patrón que aparezca en tu propio comando: espera para siempre.

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
- Tests: `pytest -m "not slow"`.
- Lint: `ruff check . && ruff format --check .`
- Ejecutar: `python main.py`.
