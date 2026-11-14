#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname -- "$(dirname -- "$SERVICE_DIR")")"
PYTHON="${AVITO_PYTHON:-$WORKSPACE_DIR/.venv/bin/python}"
export AVITO_MODEL_PATH="${AVITO_MODEL_PATH:-$WORKSPACE_DIR/models/baseline/tfidf_baseline.joblib}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$AVITO_MODEL_PATH" ]]; then
  echo "Model not found: $AVITO_MODEL_PATH" >&2
  exit 1
fi

cd -- "$SERVICE_DIR"
exec "$PYTHON" -m uvicorn app.main:app \
  --host "${AVITO_HOST:-127.0.0.1}" --port "${AVITO_PORT:-8001}"
