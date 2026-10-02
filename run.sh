#!/usr/bin/env bash
# MedPortal — start script: worker first (ClinFusion-32B resident), then the backend.
# Stop with ./stop.sh · Check status with ./status.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

WORKER_PORT="${WORKER_PORT:-8100}"
BACKEND_PORT="${BACKEND_PORT:-8080}"
# ClinFusion output budget (per answer). Long clinical answers were cut mid-sentence
# at the old default of 4096; generation still stops at EOS, so this is only a ceiling.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
SUPERVISOR_PID_FILE="logs/worker-supervisor.pid"
BACKEND_PID_FILE="logs/backend.pid"

# Clean up any previous installation (supervisor + worker + backend)
if [ -x ./stop.sh ]; then
  ./stop.sh >/dev/null 2>&1 || true
fi

# Worker supervisor: restarts the worker if it dies (survives SSH disconnects).
# The MP_WORKER_SUPERVISOR=1 marker lets stop.sh find and stop this loop safely.
setsid nohup bash -c '
  export MP_WORKER_SUPERVISOR=1
  export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
  while true; do
    echo "[worker] starting $(date)"
    /home/nvidia/miniconda3/envs/clinfusion/bin/python worker/clinfusion_worker.py >> logs/worker.log 2>&1 || true
    echo "[worker] exited; retrying in 10s"
    sleep 10
  done
' >/dev/null 2>&1 &
echo $! > "$SUPERVISOR_PID_FILE"
echo "worker supervisor started (PID $(cat "$SUPERVISOR_PID_FILE"))"

echo "Loading the 32B model (waiting for health, ~14 min; progress: tail -f logs/worker.log)…"
READY=0
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${WORKER_PORT}/health" 2>/dev/null | grep -q '"ready": *true'; then
    echo "worker is ready."
    READY=1
    break
  fi
  if [ $((i % 6)) -eq 0 ]; then
    echo "  …bekleniyor ($((i / 6)) dk) | son log: $(tail -n 1 logs/worker.log 2>/dev/null | cut -c1-100)"
  fi
  sleep 10
done
if [ "$READY" != "1" ]; then
  echo "WARNING: worker did not become ready within 40 minutes; starting the backend anyway."
fi

echo "[backend] starting…"
setsid nohup /home/nvidia/miniconda3/envs/medportal/bin/python -m uvicorn \
  backend.app.main:app --host 0.0.0.0 --port "$BACKEND_PORT" >> logs/backend.log 2>&1 &
BE_PID=$!
echo "$BE_PID" > "$BACKEND_PID_FILE"

sleep 2
if kill -0 "$BE_PID" 2>/dev/null; then
  echo "backend PID: $BE_PID"
else
  echo "WARNING: the backend may not have started; see logs/backend.log (recorded PID: $BE_PID)"
fi

echo "Open: http://192.168.1.162:${BACKEND_PORT}"
echo "Model durumu: curl -s http://127.0.0.1:${WORKER_PORT}/health"