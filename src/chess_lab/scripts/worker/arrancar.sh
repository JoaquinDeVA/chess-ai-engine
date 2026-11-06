#!/usr/bin/env bash
# Arranca roundrobin.py en ~/worker del nodo, desacoplado de la tarea de pueue (se ejecuta allí, desde la copia del repo).
# Trampas: no lo arranca si ya está en marcha; para pararlo, SIGINT al proceso (nunca kill -9: se pierde el estado de config.json).
set -euo pipefail
W="$HOME/worker"
if pgrep -f "roundrobin.py queue.json" >/dev/null; then echo "ya está en marcha"; exit 0; fi
cd "$W"
setsid nohup python3 roundrobin.py queue.json --chunk-rounds 100 --concurrency 4 >>roundrobin.log 2>&1 </dev/null &
sleep 3
pgrep -af "roundrobin.py|fastchess" | cut -c1-120
