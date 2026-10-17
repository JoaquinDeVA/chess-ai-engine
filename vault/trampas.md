---
tipo: trampas
---
# Trampas conocidas

Solo lo que ya costó tiempo. Formato: síntoma → causa → qué hacer.

## Subagentes y tareas de fondo

- **Un subagente dice «tests pasan» o «sincronizado» y no es verdad** → su palabra no es verificación → ejecuta tú el test, el lint o el grep y mira la salida antes de decirlo.
- **`until pgrep -f "patron"` espera para siempre** → el patrón aparece en la propia línea de comandos del bucle → usa un fichero de estado o `timeout`, nunca `pgrep -f` con un texto que tú mismo escribes.
- **Dos subagentes sobre los mismos ficheros** → conflictos y ediciones que se pisan → un subagente por conjunto de ficheros, o secuencial.
- **Subagente con modelo intermedio para un encargo guiado** → más coste y más fallos → `haiku` por defecto; `sonnet` solo para juicio.
- **Brief sin rutas exactas** → el subagente improvisa y cambia ficheros que no tocaba → rutas, comandos y criterio de aceptación en el brief.
- **Comando de debug que no corre** (backticks dentro de un heredoc) → el script no existe y parece que «tarda» → escribe el script con la herramienta de escritura en el scratchpad y ejecútalo.
- **`rm -f $VAR/*` bloqueado por seguridad** → la variable puede quedar vacía → usa la ruta literal.
- **Un cambio de 2 modelos tardó 27 min** → el subagente ejecutó `pytest -m slow` (pools de torneo) en lugar del comando del brief, y quedaron procesos hijos huérfanos de ejecuciones anteriores → el brief lleva el comando exacto y el criterio «no ejecutes nada más»; mira `ps` tras cada subagente y mata los huérfanos.

## Tablas y documentación generada

- **`docs/training/MODELS_HISTORY.md` mostraba modelos ya borrados** → su tabla la regenera `comparison/report.py` desde las fichas de `models/cards/`; editarla a mano se pierde, y la fuente seguía teniendo los modelos → cambia la fuente y regenera; no edites la tabla.
- **«Sincronizado» en la documentación sin comprobarlo** → afirmación sin verificación → cada cambio de nombre o de comando se verifica con grep sobre todo el repo antes de darlo por hecho.
