#!/usr/bin/env bash
# MedPortal — stop script.
# Order matters: stop the worker supervisor loop first (otherwise the worker is
# respawned within 10 s), then the worker, then the backend. Ports are verified at the end.
#
# Usage: ./stop.sh [--force]
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

WORKER_PORT="${WORKER_PORT:-8100}"
BACKEND_PORT="${BACKEND_PORT:-8080}"
SUPERVISOR_PID_FILE="logs/worker-supervisor.pid"
BACKEND_PID_FILE="logs/backend.pid"

port_busy() {
  ss -ltn 2>/dev/null | grep -q ":$1 "
}

wait_port_free() {
  local port="$1"
  for _ in $(seq 1 24); do
    port_busy "$port" || return 0
    sleep 0.25
  done
  return 1
}

# SIGTERM to the pid and its children; SIGKILL after 5 s
kill_tree() {
  local pid="$1"
  kill -TERM "-$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
  pkill -TERM -P "$pid" 2>/dev/null || true
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    if [ "$FORCE" = "1" ]; then
      kill -KILL "-$pid" 2>/dev/null || true
      kill -KILL "$pid" 2>/dev/null || true
      pkill -KILL -P "$pid" 2>/dev/null || true
    fi
  fi
}

echo "== MedPortal durduruluyor =="

# 1) Worker supervisor loop FIRST — it would respawn the worker otherwise
if [ -f "$SUPERVISOR_PID_FILE" ]; then
  SUP_PID="$(cat "$SUPERVISOR_PID_FILE" 2>/dev/null)"
  if [ -n "$SUP_PID" ] && kill -0 "$SUP_PID" 2>/dev/null; then
    echo "supervisor durduruluyor (PID $SUP_PID)"
    kill_tree "$SUP_PID"
  fi
  rm -f "$SUPERVISOR_PID_FILE"
fi
# Fallback: catch the loop by its marker if the pid file was lost
if pgrep -f "MP_WORKER_SUPERVISOR=1" >/dev/null 2>&1; then
  echo "supervisor (marker) durduruluyor"
  pkill -TERM -f "MP_WORKER_SUPERVISOR=1" 2>/dev/null || true
  sleep 1
  [ "$FORCE" = "1" ] && pkill -KILL -f "MP_WORKER_SUPERVISOR=1" 2>/dev/null || true
fi

# 2) Worker (ClinFusion-32B)
if pgrep -f "clinfusion_worker.py" >/dev/null 2>&1; then
  echo "stopping worker (32B memory is released)"
  pkill -TERM -f "clinfusion_worker.py" 2>/dev/null || true
  sleep 3
  if pgrep -f "clinfusion_worker.py" >/dev/null 2>&1; then
    if [ "$FORCE" = "1" ]; then
      pkill -KILL -f "clinfusion_worker.py" 2>/dev/null || true
    else
      echo "  worker is still shutting down; use --force to SIGKILL"
    fi
  fi
fi

# 3) Backend (uvicorn)
if [ -f "$BACKEND_PID_FILE" ]; then
  BE_PID="$(cat "$BACKEND_PID_FILE" 2>/dev/null)"
  if [ -n "$BE_PID" ] && kill -0 "$BE_PID" 2>/dev/null; then
    echo "backend durduruluyor (PID $BE_PID)"
    kill_tree "$BE_PID"
  fi
  rm -f "$BACKEND_PID_FILE"
fi
if pgrep -f "[u]vicorn.*backend.app.main" >/dev/null 2>&1; then
  echo "stopping backend (leftover)"
  pkill -TERM -f "[u]vicorn.*backend.app.main" 2>/dev/null || true
  sleep 1
  [ "$FORCE" = "1" ] && pkill -KILL -f "[u]vicorn.*backend.app.main" 2>/dev/null || true
fi

# 4) Verify
OK=1
if wait_port_free "$WORKER_PORT"; then
  echo "port $WORKER_PORT is free"
else
  echo "UYARI: port $WORKER_PORT hâlâ dolu"; OK=0
fi
if wait_port_free "$BACKEND_PORT"; then
  echo "port $BACKEND_PORT is free"
else
  echo "UYARI: port $BACKEND_PORT hâlâ dolu"; OK=0
fi

if pgrep -f "clinfusion_worker.py|MP_WORKER_SUPERVISOR=1|[u]vicorn.*backend.app.main" >/dev/null 2>&1; then
  echo "WARNING: MedPortal processes are still running (try ./stop.sh --force)"
  pgrep -af "clinfusion_worker.py|MP_WORKER_SUPERVISOR=1|[u]vicorn.*backend.app.main" | sed 's/^/  /'
  OK=0
fi

if [ "$OK" = "1" ]; then
  echo "== DURDURULDU =="
else
  echo "== Partially stopped (see warnings above) =="
fi