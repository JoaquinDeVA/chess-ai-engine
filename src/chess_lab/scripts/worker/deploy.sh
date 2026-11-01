#!/usr/bin/env bash
# Copia al trabajador el binario UCI, fastchess y las aperturas.
# Uso: deploy.sh   (HOST y PORT se pueden sobrescribir por entorno)
# Se compila en el portátil (glibc 2.39) y se copia: el trabajador tiene glibc 2.43 y no tiene Rust.
set -euo pipefail

source "$(dirname "$0")/env.sh"
FASTCHESS="${FASTCHESS:-$HOME/fastchess/fastchess}"

ssh -p "$PORT" "$HOST" 'mkdir -p ~/worker/results'
scp -P "$PORT" "$ROOT/rust/target/release/uci" "$FASTCHESS" "$ROOT/data/openings/openings.epd" "$HOST:worker/"
ssh -p "$PORT" "$HOST" 'chmod +x ~/worker/uci ~/worker/fastchess && ~/worker/fastchess -version | head -n 1'
