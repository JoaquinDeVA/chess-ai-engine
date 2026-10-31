#!/usr/bin/env bash
# Vigila la conexión con el trabajador desde el portátil y la recupera si se cae.
# Uso: keepalive.sh   (HOST, PORT, WIFI, CHECK y RETRY se pueden sobrescribir por entorno)
# Cada CHECK segundos comprueba el SSH; si falla, cada RETRY segundos apaga y enciende el wifi
# y reselecciona la conexión WIFI hasta que el SSH responda. Registro en data/matches/keepalive.log.
# Para que el portátil no se suspenda: systemd-inhibit --what=sleep keepalive.sh
set -u

HOST="${HOST:-devicente@192.168.1.124}"
PORT="${PORT:-2222}"
WIFI="${WIFI:-comprateunawifisorata}"
CHECK="${CHECK:-1800}"
RETRY="${RETRY:-300}"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
LOG="$ROOT/data/matches/keepalive.log"
mkdir -p "$(dirname "$LOG")"

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
alive() { ssh -o ConnectTimeout=8 -o BatchMode=yes -p "$PORT" "$HOST" true 2>/dev/null; }

log "inicio: comprobando $HOST cada ${CHECK}s"
while true; do
    if alive; then
        log "ok"
        sleep "$CHECK"
        continue
    fi
    log "sin conexión con el trabajador"
    until alive; do
        log "reiniciando wifi y reseleccionando $WIFI"
        nmcli radio wifi off
        sleep 5
        nmcli radio wifi on
        sleep 10
        nmcli connection up "$WIFI" >/dev/null 2>&1 || log "nmcli no pudo activar $WIFI"
        sleep 10
        alive && break
        log "sigue sin conexión; reintento en ${RETRY}s"
        sleep "$RETRY"
    done
    log "conexión recuperada"
done
