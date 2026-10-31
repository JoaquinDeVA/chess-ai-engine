#!/usr/bin/env bash
# Juega un match de fastchess en el trabajador y trae el PGN y el registro.
# Uso: match.sh NOMBRE [opciones de fastchess...]
# Las opciones por defecto (motores, aperturas, PGN) se añaden antes; las del usuario las pisan.
# Resultado: data/matches/NOMBRE/{match.pgn,match.log}
set -euo pipefail

HOST="${HOST:-devicente@192.168.1.124}"
PORT="${PORT:-2222}"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
NAME="${1:?uso: match.sh NOMBRE [opciones de fastchess...]}"
shift

ssh -p "$PORT" "$HOST" bash -s -- "$NAME" "$@" <<'REMOTE'
set -euo pipefail
name="$1"; shift
dir=~/worker/results/"$name"
mkdir -p "$dir"
cd ~/worker
./fastchess -engine cmd=./uci name=A -engine cmd=./uci name=B \
    -openings file=openings.epd format=epd order=random \
    -pgnout file="$dir/match.pgn" "$@" 2>&1 | tee "$dir/match.log"
REMOTE

mkdir -p "$ROOT/data/matches/$NAME"
scp -P "$PORT" "$HOST:worker/results/$NAME/match.*" "$ROOT/data/matches/$NAME/"
