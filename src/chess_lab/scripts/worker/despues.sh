#!/usr/bin/env bash
# Encadena en el nodo lo que hay que hacer cuando termina el entrenamiento, sin intervención (se ejecuta allí, desde ~/worker;
# lo lanza arrancar_despues.sh). Pasos, cada uno cuando se cumple su condición:
#   1. Espera a que acabe entrenar.sh. Copia a ~/worker/models los JSON de data/results/entrenar/e30/uci_models/ y, si están
#      los tres modelos, añade a queue.json los trabajos de queue_despues.json (los SPRT de las redes nuevas; el round robin
#      relee queue.json en cada turno).
#   2. Espera a que no quede ningún round robin ni entrenamiento (el nodo libre) y entonces juega el 10 min + 5 s
#      (queue_10min.json) él solo, con el nodo entero para los relojes.
# Trampas: no se duplica (flock sobre despues.lock; si ya corre, sale); los pasos son idempotentes, así que se puede relanzar. Si el entrenamiento falla
# y faltan modelos, no añade los SPRT de redes y lo deja dicho en despues.log. El 10 min + 5 s puede tardar en empezar tanto
# como tarde en vaciarse la cola del round robin (hasta 6000 rondas por trabajo): para saltarse la espera, parar el round robin
# con parar.sh (SIGINT) y este script lo lanza al instante.
set -uo pipefail
W="$HOME/worker"
E30="$HOME/projects/07-ChessAIEngine/data/results/entrenar/e30/uci_models"
cd "$W"
log() { echo "$(date '+%F %T') $*" >> despues.log; }
exec 9> despues.lock
flock -n 9 || { echo "ya está en marcha"; exit 0; }

log "esperando a que termine entrenar.sh"
while pgrep -f "[e]ntrenar.sh" > /dev/null; do sleep 120; done

if ls "$E30"/net_pst_k4_w64.json "$E30"/net_pst_w64.json "$E30"/net_pst_w64x16.json > /dev/null 2>&1; then
    cp "$E30"/*.json models/
    python3 - <<'EOF'
import json
q = json.load(open("queue.json"))
nuevos = json.load(open("queue_despues.json"))["jobs"]
tengo = {j["name"] for j in q["jobs"]}
q["jobs"] += [j for j in nuevos if j["name"] not in tengo]
json.dump(q, open("queue.json.tmp", "w"), indent=1)
EOF
    mv queue.json.tmp queue.json
    log "modelos e30 copiados y SPRT de redes añadidos a queue.json"
else
    log "FALTAN modelos e30 en $E30: no se añaden los SPRT de redes (¿falló entrenar.sh?)"
fi

log "esperando a que el nodo esté libre para el 10 min + 5 s"
while pgrep -f "[r]oundrobin.py|[c]hess_lab.training" > /dev/null; do sleep 300; done
log "nodo libre: empieza el 10 min + 5 s"
python3 roundrobin.py queue_10min.json --chunk-rounds 50 --concurrency 4 >> roundrobin_10min.log 2>&1
log "10 min + 5 s terminado (ver results/model_net_vs_linear_10min/state.json)"
