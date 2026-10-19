#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
if [[ -f deploy/vercel/.env.local ]]; then
  source deploy/vercel/.env.local
fi
source .env
set +a
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
exec .venv/bin/python -m uvicorn src.local_search_app:app --host 127.0.0.1 --port 8080
