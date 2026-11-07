#!/usr/bin/env bash
# Launch Dumbocracy v2: Python engine (FastAPI, :61008) + Next.js web game (:61007).
#   scripts/run_game.sh            # production: next build (if needed) + next start
#   scripts/run_game.sh dev        # development: next dev (hot reload)
# Open http://127.0.0.1:61007  (over SSH: ssh -L 61007:127.0.0.1:61007 <host>)
set -euo pipefail
MODE="${1:-prod}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_PORT="${ENGINE_PORT:-61008}"
WEB_PORT=61007
export DUMBOCRACY_API="http://127.0.0.1:${ENGINE_PORT}"
# The host is shared and heavily loaded: oversubscribed CPU torch threads make electorate generation
# (canonical CPU streams, R8) ~10x slower. Keep CPU thread pools small.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

cleanup() { [[ -n "${ENGINE_PID:-}" ]] && kill "$ENGINE_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[dumbocracy] engine  -> ${DUMBOCRACY_API}"
conda run --no-capture-output -n mas python -m uvicorn dumbocracy.api.app:app \
  --host 127.0.0.1 --port "${ENGINE_PORT}" --log-level warning &
ENGINE_PID=$!

cd "${ROOT}/web"
if [[ "${MODE}" == "dev" ]]; then
  echo "[dumbocracy] web dev -> http://127.0.0.1:${WEB_PORT}"
  conda run --no-capture-output -n mas npm run dev
else
  [[ -d .next ]] || conda run --no-capture-output -n mas npm run build
  echo "[dumbocracy] web     -> http://127.0.0.1:${WEB_PORT}"
  conda run --no-capture-output -n mas npm run start
fi
