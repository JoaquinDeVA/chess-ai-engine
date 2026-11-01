# Trabajador y raíz del repo, compartidos por deploy.sh y match.sh (se carga con `source`).
# HOST y PORT se pueden sobrescribir por entorno.
HOST="${HOST:-devicente@192.168.1.124}"
PORT="${PORT:-2222}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
