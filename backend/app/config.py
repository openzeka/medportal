import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "workspace"
UPLOADS = WORKSPACE / "uploads"
JOBS = WORKSPACE / "jobs"
FRONTEND = ROOT / "frontend"

MEDPORTAL_PY = os.environ.get("MEDPORTAL_PY", "/home/nvidia/miniconda3/envs/medportal/bin/python")
RADAR_PY = os.environ.get("RADAR_PY", "/home/nvidia/miniconda3/envs/radar/bin/python")
CLINFUSION_PY = os.environ.get("CLINFUSION_PY", "/home/nvidia/miniconda3/envs/clinfusion/bin/python")
RADAR_SCRIPT = os.environ.get(
    "RADAR_SCRIPT", "/home/nvidia/radar/RADAR_inference/inference_demo.py"
)
CLINFUSION_REPO = os.environ.get("CLINFUSION_REPO", "/home/nvidia/ClinFusion")
CLINFUSION_MODEL = os.environ.get("CLINFUSION_MODEL", "cache/models/ClinFusion-32B")

WORKER_HOST = os.environ.get("WORKER_HOST", "127.0.0.1")
WORKER_PORT = int(os.environ.get("WORKER_PORT", "8100"))
WORKER_URL = f"http://{WORKER_HOST}:{WORKER_PORT}"

MAX_UPLOAD_MB = 500
RADAR_TIMEOUT = 600
CHAT_TIMEOUT = 1800
ALLOWED_NIFTI = (".nii.gz", ".nii")
ALLOWED_IMAGE = (".jpg", ".jpeg", ".png")


def ensure_dirs() -> None:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
