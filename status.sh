#!/usr/bin/env bash
# MedPortal — status report: processes, ports, worker health, backend api/status.
set -uo pipefail
cd "$(dirname "$0")"

WORKER_PORT="${WORKER_PORT:-8100}"
BACKEND_PORT="${BACKEND_PORT:-8080}"

proc_line() {
  local pat="$1"
  local pids
  pids="$(pgrep -f "$pat" 2>/dev/null | tr '\n' ' ')"
  if [ -n "${pids// /}" ]; then
    echo "running (PID: $pids)"
  else
    echo "down"
  fi
}

echo "== MedPortal durum =="
echo "worker supervisor: $(proc_line "MP_WORKER_SUPERVISOR=1")"
echo "worker (32B)     : $(proc_line "clinfusion_worker.py")"
echo "backend          : $(proc_line "[u]vicorn.*backend.app.main")"

for p in "$WORKER_PORT" "$BACKEND_PORT"; do
  if ss -ltn 2>/dev/null | grep -q ":$p "; then
    echo "port $p        : dinliyor"
  else
    echo "port $p        : free"
  fi
done

echo "--- worker /health (:$WORKER_PORT) ---"
curl -s -m 3 "http://127.0.0.1:${WORKER_PORT}/health" || echo "(unreachable)"
echo
echo "--- backend /api/status (:$BACKEND_PORT) ---"
curl -s -m 3 "http://127.0.0.1:${BACKEND_PORT}/api/status" || echo "(unreachable)"
echo