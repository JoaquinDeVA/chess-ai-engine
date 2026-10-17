# Base de evaluaciones de Lichess

## Qué es
Base pública `lichess_db_eval.jsonl.zst` (CC0): 416.442.401 posiciones evaluadas con Stockfish, en
JSON Lines comprimido (22,4 GB). Es la fuente de todos los datos de entrenamiento. Detalle en
`docs/training/DATA.md`.

## Restricciones
- **Sin orden aleatorio:** las primeras líneas son más profundas, con más mates y más equilibradas.
  Las primeras 1M no son una muestra uniforme. Usamos las primeras ~124,9 M líneas.
- **Sin contadores de jugadas en el FEN.**
- **Varias evaluaciones por posición** (`evals`), a distintas profundidades; el conversor elige la de
  mayor profundidad.
- **Signo:** el punto de vista de `cp` y `mate` es el de las blancas (detectado en la prueba).
- **HTTP 429** si se abren demasiadas conexiones a la vez: con 16 nos rechazaron, usamos 4 con `aria2c`.
- **Red lenta** en esta conexión (~1 MB/s): la descarga se reanuda con aria2c y luego se convierte desde
  el fichero local, nunca en streaming.
- **Valores acotados:** `cp` llega a ±20000 (acotado).
