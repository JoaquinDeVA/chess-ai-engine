#!/usr/bin/env bash
# Lanza matches de fastchess en el trabajador sin depender de esta conexión.
# Uso: match.sh start NOMBRE [opciones de fastchess...]   lanza (o reanuda si ya hay estado) y vuelve
#      match.sh status NOMBRE                              ¿corre?, y resumen del log
#      match.sh fetch NOMBRE                               trae match.pgn y match.log a data/matches/NOMBRE/
#      match.sh stop NOMBRE                                para fastchess (el estado queda guardado para reanudar)
# A_ARGS y B_ARGS añaden opciones solo a un motor (p. ej. A_ARGS='nodes=20000 option.Techniques=lmr').
# Reanudar = repetir el mismo `start NOMBRE`: carga results/NOMBRE/config.json, que fastchess guarda solo.
# fastchess escribe config.json solo al salir limpio o con SIGINT: parar siempre con `stop`, nunca `kill -9`
# (se pierde el estado del SPRT; las partidas del PGN siguen valiendo).
# Un nombre de técnica mal escrito no para nada: el binario responde `info string unknown technique` y sigue con
# las de serie, y fastchess no lo muestra; probar antes la lista con `setoption` a mano.
# Si el trabajador no responde tras reiniciar el sobremesa: la IP de WSL2 cambia en cada arranque y hay que
# rehacer en Windows el `netsh interface portproxy` (2222 -> IP de WSL, puerto 22). sshd escucha por ssh.socket,
# así que tocar el puerto en sshd_config no hace nada.
set -euo pipefail

source "$(dirname "$0")/env.sh"
CMD="${1:?uso: match.sh start|status|fetch|stop NOMBRE [opciones de fastchess...]}"
NAME="${2:?falta NOMBRE}"
shift 2
REMOTE_DIR="worker/results/$NAME"

case "$CMD" in
start)
    ssh -p "$PORT" "$HOST" A_ARGS="${A_ARGS:-}" B_ARGS="${B_ARGS:-}" bash -s -- "$NAME" "$@" <<'REMOTE'
set -euo pipefail
name="$1"; shift
dir=~/worker/results/"$name"
mkdir -p "$dir"
cd ~/worker
if pgrep -f "[r]esults/$name/" >/dev/null; then echo "ya corre: $name"; exit 1; fi
if [ -f "$dir/config.json" ]; then
    echo "reanudando $name desde $dir/config.json"
    args=(-config file="$dir/config.json" outname="$dir/config.json")
else
    args=(-engine cmd=./uci name=A $A_ARGS -engine cmd=./uci name=B $B_ARGS
          -openings file=openings.epd format=epd order=random
          -pgnout file="$dir/match.pgn" "$@" -config outname="$dir/config.json")
fi
setsid nohup ./fastchess "${args[@]}" >>"$dir/match.log" 2>&1 </dev/null &
echo "lanzado $name (pid $!)"
REMOTE
    ;;
status)
    ssh -p "$PORT" "$HOST" bash -s -- "$NAME" <<'REMOTE'
name="$1"; dir=~/worker/results/"$name"
pgrep -f "[r]esults/$name/" >/dev/null && echo "corriendo" || echo "parado"
echo "partidas terminadas: $(grep -c '^Finished game' "$dir/match.log")"
grep -v '^Warning' "$dir/match.log" | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'Elo:|Games:|Ptnml|LLR|SPRT' | tail -5
REMOTE
    ;;
fetch)
    mkdir -p "$ROOT/data/matches/$NAME"
    scp -P "$PORT" "$HOST:$REMOTE_DIR/match.*" "$HOST:$REMOTE_DIR/config.json" "$ROOT/data/matches/$NAME/"
    ;;
stop)
    ssh -p "$PORT" "$HOST" "pkill -INT -f '[r]esults/$NAME/' || echo 'no corría'"
    ;;
*) echo "orden desconocida: $CMD" >&2; exit 2 ;;
esac
