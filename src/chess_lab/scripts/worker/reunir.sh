#!/usr/bin/env bash
# Reúne en data/results/roundrobin/ (de la copia del repo en el nodo) el estado de los SPRT de ~/worker para traerlo
# con «fetch». Solo copia llr.csv, state.json y el log del planificador, no los PGN.
set -euo pipefail
W="$HOME/worker"
D=data/results/roundrobin
rm -rf "$D" && mkdir -p "$D"
cp "$W/roundrobin.log" "$D/" 2>/dev/null || true
for dir in "$W"/results/*/; do
    name=$(basename "$dir")
    mkdir -p "$D/$name"
    cp "$dir/llr.csv" "$dir/state.json" "$D/$name/" 2>/dev/null || true
done
ls "$D"
