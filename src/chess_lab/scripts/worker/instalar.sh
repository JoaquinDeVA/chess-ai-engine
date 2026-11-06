#!/usr/bin/env bash
# Instala en ~/worker del nodo lo que sube la tarea «subir» de .computo (se ejecuta allí, desde la copia del repo).
# Trampas: el binario se sustituye con mv (atómico) para no romper los fastchess en marcha; roundrobin.py no se
# reinicia (si ha cambiado, solo avisa) y relee queue.json en cada turno, así que los trabajos nuevos entran solos.
set -euo pipefail
W="$HOME/worker"
mkdir -p "$W/models"
cp rust/target/release/uci "$W/uci.new" && mv "$W/uci.new" "$W/uci"
cp data/results/uci_models/*.json "$W/models/"
cp data/openings/openings.epd "$W/"
cp src/chess_lab/scripts/worker/queue.json "$W/queue.json"
cmp src/chess_lab/scripts/worker/roundrobin.py "$W/roundrobin.py" || echo "AVISO: roundrobin.py cambió; se aplica al reiniciarlo"
pgrep -af "roundrobin.py" || echo "AVISO: roundrobin no está en marcha"
printf 'setoption name Model value %s/models/net_hybrid_w64.json\nposition startpos\neval\nquit\n' "$W" | "$W/uci"
