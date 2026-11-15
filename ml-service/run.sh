#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname -- "$SERVICE_DIR")"
PYTHON="${AVITO_PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$REPO_DIR/models/c_syntax/manifest.json" ]]; then
  echo "C_syntax bundle not found: $REPO_DIR/models/c_syntax" >&2
  exit 1
fi

cd -- "$REPO_DIR"
exec "$PYTHON" -m uvicorn app.main:app \
  --app-dir "$SERVICE_DIR" --host "${AVITO_HOST:-127.0.0.1}" --port "${AVITO_PORT:-8001}"
