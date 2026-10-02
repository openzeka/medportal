#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8080}"
# Demo scan shipped with RADAR; override NII to test with your own volume.
RADAR_REPO="${RADAR_REPO:-$HOME/radar}"
NII="${NII:-${RADAR_REPO}/data/demo_cases/AC423ccbe.nii.gz}"
if [ ! -f "$NII" ]; then
  echo "Demo scan not found: $NII"
  echo "Set NII=/path/to/scan.nii.gz and re-run."
  exit 1
fi

echo "== status =="; curl -sf "$BASE/api/status"; echo
echo "== upload =="
UP=$(curl -sf -F "file=@${NII}" "$BASE/api/upload")
echo "$UP"
UPLOAD_ID=$(echo "$UP" | python3 -c "import sys,json;print(json.load(sys.stdin)['upload_id'])")
echo "== slice =="; curl -sf -o /tmp/smoke_slice.png -D /tmp/smoke_h.txt "$BASE/api/nifti/${UPLOAD_ID}/slice?idx=10"; grep -i x-total-slices /tmp/smoke_h.txt
echo "== radar =="
J=$(curl -sf -H 'Content-Type: application/json' -d "{\"upload_id\":\"${UPLOAD_ID}\"}" "$BASE/api/radar")
JID=$(echo "$J" | python3 -c "import sys,json;print(json.load(sys.stdin)['job_id'])")
DONE=0
for i in $(seq 1 120); do
  R=$(curl -sf "$BASE/api/radar/${JID}")
  ST=$(echo "$R" | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])")
  [ "$ST" = "done" ] && { DONE=1; echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);print('findings:',len(d['findings']),'top:',d['findings'][0]['name'])"; break; }
  [ "$ST" = "error" ] && { echo "RADAR error: $R"; exit 1; }
  sleep 2
done
[ "$DONE" = "1" ] || { echo "RADAR timed out"; exit 1; }
echo "== backend tests =="
cd "$(dirname "$0")/.."
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
MEDPORTAL_PY="${MEDPORTAL_PY:-$CONDA_ROOT/envs/medportal/bin/python}"
"$MEDPORTAL_PY" -m pytest backend/tests -q
echo "SMOKE OK"
