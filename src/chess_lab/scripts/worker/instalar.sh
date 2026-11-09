#!/usr/bin/env bash
# Instala en ~/worker del nodo lo que se sube al nodo (ver `.computo/tareas.json`) (se ejecuta allí, desde la copia del repo).
# Cada binario va a ~/worker/bin/<hash>/uci, con <hash> = los 12 primeros caracteres de su sha256 (en local:
# `sha256sum rust/target/release/uci | cut -c1-12`); cada trabajo de queue.json nombra el suyo en «uci».
# Trampas: un binario instalado no se sobrescribe, así que un SPRT largo sigue con el suyo aunque llegue otro;
# roundrobin.py solo se sustituye si no está en marcha (si no, avisa) y relee queue.json en cada turno, así que los
# trabajos nuevos entran solos. Los binarios anteriores a la retirada de los modelos lineales (p. ej.
# bin/1fa18f0aab94) exigen "perspective": "side_to_move" en los JSON de modelo; no reexportar
# data/results/uci_models/ sin ese campo mientras sigan en marcha.
set -euo pipefail
W="$HOME/worker"
H=$(sha256sum rust/target/release/uci | cut -c1-12)
mkdir -p "$W/models" "$W/bin/$H"
[ -f "$W/bin/$H/uci" ] || { cp rust/target/release/uci "$W/bin/$H/uci.new" && mv "$W/bin/$H/uci.new" "$W/bin/$H/uci"; }
cp data/results/uci_models/*.json "$W/models/"
cp data/openings/openings.epd "$W/"
cp src/chess_lab/scripts/worker/queue_despues.json src/chess_lab/scripts/worker/queue_10min.json src/chess_lab/scripts/worker/despues.sh "$W/"
# queue.json solo si no hay round robin: en marcha, despues.sh le añade trabajos y copiarlo encima los borraría.
[ -f "$W/queue.json" ] && pgrep -f "[r]oundrobin.py queue.json" >/dev/null || cp src/chess_lab/scripts/worker/queue.json "$W/queue.json"
if pgrep -f "[r]oundrobin.py queue.json" >/dev/null; then
    cmp src/chess_lab/scripts/worker/roundrobin.py "$W/roundrobin.py" || echo "AVISO: roundrobin.py cambió; se aplica al reiniciarlo"
else
    cp src/chess_lab/scripts/worker/roundrobin.py "$W/roundrobin.py"
    echo "AVISO: roundrobin no está en marcha"
fi
echo "binario: bin/$H/uci"
printf 'setoption name Model value %s/models/net_pst_w32.json\nposition startpos\neval\nquit\n' "$W" | "$W/bin/$H/uci"
