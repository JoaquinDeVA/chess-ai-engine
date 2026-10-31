#!/usr/bin/env bash
# Copia al trabajador el binario UCI, fastchess y las aperturas.
# Uso: deploy.sh   (HOST y PORT se pueden sobrescribir por entorno)
set -euo pipefail

HOST="${HOST:-devicente@192.168.1.124}"
PORT="${PORT:-2222}"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
FASTCHESS="${FASTCHESS:-$HOME/fastchess/fastchess}"

ssh -p "$PORT" "$HOST" 'mkdir -p ~/worker/results'
scp -P "$PORT" "$ROOT/rust/target/release/uci" "$FASTCHESS" "$ROOT/data/openings/openings.epd" "$HOST:worker/"
ssh -p "$PORT" "$HOST" 'chmod +x ~/worker/uci ~/worker/fastchess && ~/worker/fastchess -version | head -n 1'
