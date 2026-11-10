#!/usr/bin/env bash
# Bot de Lichess en el nodo (~/lichess), con el motor en su modo más fuerte, y relevo con el round robin.
# Uso: lichess.sh subir            (en local: copia bot, .env, binario, libro y tablas al nodo y lo instala)
#      lichess.sh instalar|arrancar|parar|estado   (en el nodo; con ssh, `bash ~/lichess/lichess.sh <orden>`)
# Modo más fuerte: Model net_pst_w64, Book, Tablebase, técnicas por defecto, Threads 4 (docs/techniques/lazy-smp.md: seis no ganaron a cuatro), Hash 2048.
# Relevo: arrancar lanza el bot y un supervisor (desacoplados, sin pueue). El supervisor, cuando el bot llega al límite diario
# de partidas contra bots (línea «'bot_is_rate_limited': True» del log de lichess-bot) o cuando el bot sale por su cuenta, hace
# SIGINT al bot (acaba las partidas en curso) y reanuda el round robin con arrancar.sh.
# Trampas: lichess-bot necesita Python ≥ 3.11 (el del sistema es 3.10; ver instalar); nunca kill -9; el round robin y el bot comparten CPU, así que arrancar se niega si el round robin corre; «parar»
# mata antes al supervisor para que no reanude el round robin; el token va en ~/lichess/.env (modo 600) y no se imprime.
set -euo pipefail
L="$HOME/lichess"
ARRANCAR_RR="$HOME/projects/07-ChessAIEngine/src/chess_lab/scripts/worker/arrancar.sh"
LIMITE='bot_is_rate_limited.: True'
RR='[r]oundrobin.py|[f]astchess'

bot_vivo() { [ -f "$L/bot.pid" ] && kill -0 "$(cat "$L/bot.pid")" 2>/dev/null; }

case "${1:-}" in
subir)
    R=$(git rev-parse --show-toplevel)
    N=computo@y520
    ssh -o BatchMode=yes $N 'mkdir -p ~/lichess/engine ~/lichess/knowledge'
    rsync -a --exclude .git "$R/lichess/lichess-bot/" $N:lichess/lichess-bot/
    rsync -a "$R/lichess/config.yml" $N:lichess/config.local.yml
    rsync -a --chmod=F600 "$R/lichess/.env" $N:lichess/.env
    rsync -a "$R/rust/target/release/uci" $N:lichess/engine/uci
    rsync -a "$R/data/knowledge/" $N:lichess/knowledge/
    rsync -a "$R/src/chess_lab/scripts/worker/" $N:projects/07-ChessAIEngine/src/chess_lab/scripts/worker/ --exclude queue.json
    rsync -a "$R/src/chess_lab/scripts/worker/lichess.sh" $N:lichess/lichess.sh
    ssh -o BatchMode=yes $N 'bash ~/lichess/lichess.sh instalar'
    ;;
instalar)
    cd "$L"
    # lichess-bot exige Python ≥ 3.11 (StrEnum) y el python3 del sistema es 3.10: se usa el del entorno de entrenamiento (3.12).
    [ -d .venv ] || "${PY:-$HOME/train/venv/bin/python}" -m venv .venv
    .venv/bin/pip install -q -r lichess-bot/requirements.txt
    sed -e "s#^  dir:.*#  dir: \"$L/engine\"#" \
        -e 's/^    Threads:.*/    Threads: 4/' \
        -e "s#^    Hash:.*#    Hash: 2048\n    Model: \"net_pst_w64\"\n    Book: \"$L/knowledge/opening_book/opening_book.json\"\n    Tablebase: \"$L/knowledge/endgame_tablebase\"#" \
        config.local.yml > config.yml
    printf 'uci\nsetoption name Book value %s/knowledge/opening_book/opening_book.json\nsetoption name Tablebase value %s/knowledge/endgame_tablebase\nisready\nquit\n' "$L" "$L" | ./engine/uci | grep -E "readyok|error"
    grep -E "dir:|Threads|Hash|Model|Book|Tablebase" config.yml
    ;;
arrancar)
    cd "$L"
    if pgrep -f "$RR" >/dev/null; then echo "el round robin corre: pararlo antes con parar.sh" >&2; exit 1; fi
    if bot_vivo; then echo "el bot ya está en marcha"; exit 0; fi
    [ -f lichess.log ] && mv lichess.log "lichess.$(date +%F-%H%M%S).log"
    setsid nohup bash -c 'set -a; . ./.env; set +a; exec .venv/bin/python lichess-bot/lichess-bot.py --config config.yml --logfile lichess.log' </dev/null >run.out 2>&1 &
    echo $! > bot.pid
    setsid nohup bash lichess.sh supervisor </dev/null >>super.log 2>&1 &
    echo $! > super.pid
    sleep 5
    "$0" estado
    ;;
supervisor)
    cd "$L"
    echo "$(date '+%F %T') supervisor en marcha"
    while true; do
        if ! bot_vivo; then echo "$(date '+%F %T') el bot salió: reanudo el round robin"; break; fi
        if grep -q "$LIMITE" lichess.log 2>/dev/null; then
            echo "$(date '+%F %T') límite diario: SIGINT al bot (acaba las partidas en curso)"
            kill -INT "$(cat bot.pid)"
            while bot_vivo; do sleep 15; done
            echo "$(date '+%F %T') bot parado: reanudo el round robin"
            break
        fi
        sleep 20
    done
    bash "$ARRANCAR_RR"
    ;;
parar)
    cd "$L"
    [ -f super.pid ] && kill "$(cat super.pid)" 2>/dev/null || true
    if bot_vivo; then
        kill -INT "$(cat bot.pid)"
        echo "SIGINT al bot; espera a que acaben las partidas en curso"
        for _ in $(seq 1 60); do bot_vivo || { echo "parado"; exit 0; }; sleep 10; done
        echo "sigue acabando partidas; repetir «parar» o comprobar con «estado»"
    else
        echo "el bot no estaba en marcha"
    fi
    ;;
estado)
    cd "$L"
    bot_vivo && echo "bot: en marcha (PID $(cat bot.pid))" || echo "bot: parado"
    [ -f super.pid ] && kill -0 "$(cat super.pid)" 2>/dev/null && echo "supervisor: en marcha" || echo "supervisor: parado"
    echo "partidas terminadas en este log: $(grep -c 'Game over' lichess.log 2>/dev/null || true)"
    grep -q "$LIMITE" lichess.log 2>/dev/null && echo "límite diario alcanzado" || echo "límite diario: no alcanzado"
    pgrep -f "$RR" >/dev/null && echo "round robin: en marcha" || echo "round robin: parado"
    tail -2 lichess.log 2>/dev/null | cut -c1-160
    ;;
*)
    echo "uso: lichess.sh subir|instalar|arrancar|parar|estado" >&2
    exit 2
    ;;
esac
