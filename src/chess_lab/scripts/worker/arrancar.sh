#!/usr/bin/env bash
# Arranca roundrobin.py en ~/worker del nodo, desacoplado de la tarea de pueue (se ejecuta allí, desde la copia del repo).
# También deja corriendo despues.sh (espera al entrenamiento, añade los SPRT de redes y juega el 10 min + 5 s con el nodo libre).
# Trampas: no lo arranca si ya está en marcha; para pararlo, SIGINT al proceso (nunca kill -9: se pierde el estado de config.json).
set -euo pipefail
W="$HOME/worker"
cd "$W"
setsid nohup bash despues.sh </dev/null >/dev/null 2>&1 &
sleep 1
if pgrep -f "roundrobin.py queue.json" >/dev/null; then echo "ya está en marcha"; exit 0; fi
setsid nohup python3 roundrobin.py queue.json --chunk-rounds 100 --concurrency 6 >>roundrobin.log 2>&1 </dev/null &
sleep 3
pgrep -af "roundrobin.py|fastchess|despues.sh" | cut -c1-120
