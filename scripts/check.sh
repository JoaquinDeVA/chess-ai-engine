#!/usr/bin/env bash
# Repite en local los jobs del CI (.github/workflows/ci.yml).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate

echo "== lint-python"
ruff check .
ruff format --check .

echo "== rust"
(cd rust && cargo fmt --check \
  && cargo clippy --all-targets --features python -- -D warnings \
  && cargo test)

echo "== python-tests"
pytest -m "not slow"

echo "== docs"
pytest tests/test_docs.py
