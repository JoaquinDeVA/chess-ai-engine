#!/usr/bin/env bash
# Entrena en el nodo las redes de MODELOS con la receta de effort 30 (la de data/results/sobremesa/cola_red2.sh, con
# --tag e30 en vez de copiar el .npz a <modelo>@e30), las evalúa y las exporta a JSON UCI. Se ejecuta desde la copia
# del repo que se sube al nodo (ver `.computo/tareas.json`) y deja en data/results/entrenar/<etiqueta>/ (de esa copia) las
# fichas, los JSON, los .npz y los logs para traerlos con «fetch». Uso: entrenar.sh [prueba]; «prueba» entrena solo la
# primera red con effort 1, una época y la etiqueta «prueba» (ejecución en seco del flujo entero).
# El entorno vive en ~/train y se monta solo la primera vez (cada paso se salta si ya está): uv y rustup descargados de
# sus instaladores oficiales, Python 3.12 (el del sistema es 3.10 y el proyecto pide ≥ 3.11) y TensorFlow con CUDA en
# ~/train/venv-gpu (la GTX 1050 entrena la red: medido con datos sintéticos, 1,9 µs por posición frente a 22,8 en CPU, lote
# 4096; la primera época, que codifica y crea la caché, es CPU, ~230 µs por posición, y no cambia; si no hay GPU, TensorFlow usa la CPU).
# Trampas: se entrena en ~/train/repo, no en la copia, para que otra sincronización no cambie el código a mitad; se le
# copia encima lo subido (cp, sin borrar) y se reinstala el paquete, porque evaluate y definitions usan _core. Los datos
# (data/LichessPositions/, ~2,6 GB) se enlazan desde la copia y la caché se crea en ~/train/repo/data/cache. No es un
# clon de git: las fichas salen con commit «unknown», que se corrige en local al traerlas. El codificado usa
# nproc − 2 procesos y comparte CPU con el round robin: los SPRT por tiempo juegan más despacio mientras dura (a los dos
# lados por igual); los de nodos fijos no cambian.
set -euo pipefail
COPIA=$(pwd)
MODELOS=(net_pst_k4_w64 net_pst_w64 net_pst_w64x16)
RECETA=(--effort 30 --epochs 40 --patience 3 --lr-decay 0.5)
TAG=e30
if [ "${1:-}" = prueba ]; then
    MODELOS=("${MODELOS[0]}")
    RECETA=(--effort 1 --epochs 1)
    TAG=prueba
fi
OUT="$COPIA/data/results/entrenar/$TAG"
rm -rf "$OUT" && mkdir -p "$OUT/logs" "$OUT/cards" "$OUT/uci_models" "$OUT/artifacts"
# Corre un paso con su log; si falla, enseña el final del log.
corre() { local log="$OUT/logs/$1.log"; shift; "$@" > "$log" 2>&1 || { tail -40 "$log"; exit 1; }; }

T="$HOME/train"
mkdir -p "$T/bin"
cat > "$T/env.sh" <<'EOF'
export TRAIN=$HOME/train
export CARGO_HOME=$TRAIN/cargo RUSTUP_HOME=$TRAIN/rustup UV_CACHE_DIR=$TRAIN/uvcache UV_PYTHON_INSTALL_DIR=$TRAIN/python
export PATH=$TRAIN/venv-gpu/bin:$TRAIN/cargo/bin:$TRAIN/bin:$PATH
export TF_CPP_MIN_LOG_LEVEL=2
EOF
source "$T/env.sh"
command -v uv > /dev/null || corre uv sh -c 'curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$TRAIN/bin" UV_NO_MODIFY_PATH=1 sh'
command -v cargo > /dev/null || corre rustup sh -c 'curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal'
[ -x "$T/venv-gpu/bin/python" ] || corre venv uv venv --python 3.12 "$T/venv-gpu"

R="$T/repo"
mkdir -p "$R/rust" "$R/data"
cp -a src pyproject.toml README.md "$R/"
cp -a rust/src rust/models rust/Cargo.toml rust/Cargo.lock "$R/rust/"
ln -sfn "$COPIA/data/LichessPositions" "$R/data/LichessPositions"
cd "$R"
corre tensorflow uv pip install --python "$T/venv-gpu/bin/python" "tensorflow[and-cuda]"
corre instalar uv pip install --python "$T/venv-gpu/bin/python" -e ".[ml,data]"
S=src/chess_lab/models/store
for m in "${MODELOS[@]}"; do
    corre "train_$m" python -m chess_lab.training.train --model "$m" "${RECETA[@]}" --tag "$TAG"
    corre "evaluate_$m" python -m chess_lab.measure.evaluate --model "$m@$TAG"
    corre "export_$m" python -m chess_lab.scripts.export_uci_model --model "$m@$TAG" --out "$OUT/uci_models"
    cp "$S/cards/$m@$TAG.json" "$OUT/cards/"
    cp "$S/artifacts/$m@$TAG.npz" "$OUT/artifacts/"
    echo "hecho: $m@$TAG"
done
ls -l "$OUT"/*
