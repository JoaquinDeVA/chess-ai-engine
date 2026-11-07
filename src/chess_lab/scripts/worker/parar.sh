#!/usr/bin/env bash
# Para roundrobin.py en el nodo con SIGINT: acaba el turno de fastchess y guarda el estado de los SPRT en config.json.
# Trampas: nunca kill -9 (se pierde ese estado); el patrón lleva «[r]» para que pkill no se encuentre a sí mismo;
# se puede reanudar con la tarea «arrancar».
set -euo pipefail
if ! pgrep -f "[r]oundrobin.py queue.json" >/dev/null; then echo "no está en marcha"; exit 0; fi
pkill -INT -f "[r]oundrobin.py queue.json"
for _ in $(seq 1 30); do
    pgrep -f "[r]oundrobin.py queue.json" >/dev/null || { echo "parado"; exit 0; }
    sleep 2
done
echo "sigue parándose (termina el turno en curso); comprobar con «reunir»"
