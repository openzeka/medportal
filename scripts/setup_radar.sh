#!/usr/bin/env bash
# MedPortal — RADAR runtime environment setup for DGX Spark (Linux aarch64).
#
# Why this script exists instead of the upstream README's
# `pip install -r requirements.txt` steps (documented in more detail in
# README.md § 2 — three real issues surface on a fresh DGX Spark):
#
#  1. CONDA TERMS OF SERVICE BLOCK (problem #5A)
#     Conda 26.x refuses `conda create` until its default channels have had
#     their Terms of Service accepted. This comes up as
#     "CondaToSNonInteractiveError: Terms of Service have not been accepted".
#     This script auto-accepts them for the standard `defaults` channels before
#     environment creation.
#
#  2. `decord` IS NOT AVAILABLE ON aarch64 (problem #5B)
#     requirements.txt lists `decord`, which cannot be resolved on
#     aarch64 (PyPI has no matching distribution there). This does not actually
#     matter: `RADAR_inference/inference_demo.py` never imports decord — it's
#     used by the training/eval code paths. Same story for a handful of other
#     requirements.txt entries (diffusers, fairscale, timm, spacy, streamlit,
#     webdataset, opencv-python-headless==4.5.5.64, pycocotools, ...). This
#     script installs only the packages that inference_demo.py actually
#     imports (verified empirically on .161), so the install does not die on a
#     training-only dependency.
#
#  3. `conda activate` DOES NOT WORK IN NON-INTERACTIVE SHELL (problem #5C)
#     Running `conda activate radar` in a script fails with
#     "CondaError: Run 'conda init' before 'conda activate'". This script calls
#     the environment's interpreter by absolute path instead, which also keeps
#     it consistent with MedPortal's `RADAR_PY = $CONDA_ROOT/envs/radar/bin/python`.
#
# Usage:
#   ./scripts/setup_radar.sh                  # create env + install deps + download checkpoints
#   PYTHON_VERSION=3.10 ./scripts/setup_radar.sh
#   RADAR_REPO=/elsewhere/radar ./scripts/setup_radar.sh
#   ./scripts/setup_radar.sh --skip-checkpoints    # env only (skip the ~6GB download)
#   ./scripts/setup_radar.sh --dry-run              # show resolved config only

set -euo pipefail

DRY_RUN=0
SKIP_CHECKPOINTS=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --skip-checkpoints) SKIP_CHECKPOINTS=1 ;;
    esac
done

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
RADAR_REPO="${RADAR_REPO:-$HOME/radar}"
RADAR_PY="${RADAR_PY:-$CONDA_ROOT/envs/radar/bin/python}"

CONDA_BIN="$CONDA_ROOT/bin/conda"

log()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"

if [[ ! -d "$RADAR_REPO" ]]; then
    die "RADAR repo not found at $RADAR_REPO. Clone it first, or set RADAR_REPO."
fi
[[ -f "$RADAR_REPO/RADAR_inference/inference_demo.py" ]] \
    || die "$RADAR_REPO does not look like a RADAR checkout (RADAR_inference/inference_demo.py missing)"
[[ -f "$RADAR_REPO/download_scripts/download_checkpoints.py" ]] \
    || die "$RADAR_REPO does not contain download_scripts/download_checkpoints.py"

log "Platform           : $(uname -m)"
log "Target interpreter : $RADAR_PY"
log "RADAR repo         : $RADAR_REPO"
log "Checkpoints target : $RADAR_REPO/ckpt"
[[ "$SKIP_CHECKPOINTS" == "1" ]] && log "(checkpoint download disabled via --skip-checkpoints)"

if [[ "$DRY_RUN" == "1" ]]; then
    log "--dry-run: nothing else was done."
    exit 0
fi

# --- Conda Terms of Service (Conda 26.x refuses env creation until accepted) ----
if [[ ! -x "$CONDA_BIN" ]]; then
    die "conda not found at $CONDA_BIN. Install Miniconda first (see README § 1)."
fi
log "Accepting conda channel Terms of Service (required by Conda 26.x)..."
"$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 \
    || warn "could not accept ToS for pkgs/main (may already be accepted)"
"$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 \
    || warn "could not accept ToS for pkgs/r (may already be accepted)"

# --- Environment creation -------------------------------------------------------
if [[ -x "$RADAR_PY" ]]; then
    log "Interpreter already exists at $RADAR_PY, reusing it."
else
    log "Creating conda environment at $(dirname "$(dirname "$RADAR_PY")")..."
    "$CONDA_BIN" create -y -p "$(dirname "$(dirname "$RADAR_PY")")" "python=$PYTHON_VERSION"
fi

[[ -x "$RADAR_PY" ]] || die "interpreter not created as expected: $RADAR_PY"

# --- Runtime dependencies actually imported by RADAR_inference/inference_demo.py -
# Verified against inference_demo.py's imports on the real .161 run. This list
# intentionally excludes decord, diffusers, fairscale, timm, spacy, streamlit,
# webdataset, pycocotools etc. from upstream requirements.txt — those belong to
# the training and evaluation code paths and are not used by the inference
# entry point MedPortal calls. Some of them (notably decord) also have no
# aarch64 wheels at all.
log "Installing RADAR runtime dependencies..."
"$RADAR_PY" -m pip install --upgrade pip
"$RADAR_PY" -m pip install \
    "torch" "torchvision" \
    "transformers==4.25" \
    monai \
    SimpleITK \
    pandas \
    tqdm \
    huggingface_hub

# --- Checkpoints (~6 GB, via upstream's own download script) ---------------------
if [[ "$SKIP_CHECKPOINTS" == "1" ]]; then
    log "Skipping checkpoint download (--skip-checkpoints)."
else
    if [[ -f "$RADAR_REPO/ckpt/bert-base-chinese/config.json" ]]; then
        log "Checkpoints already present at $RADAR_REPO/ckpt, skipping download."
    else
        log "Downloading RADAR checkpoints (~6 GB)..."
        ( cd "$RADAR_REPO/download_scripts" && "$RADAR_PY" download_checkpoints.py )
    fi
fi

# --- Smoke check: can we import what inference_demo.py will need at runtime? ---
log "Verifying imports..."
"$RADAR_PY" - <<'PY'
import importlib, sys

required = [
    ("torch",              "torch"),
    ("torchvision",        "torchvision"),
    ("transformers",       "transformers"),
    ("monai",              "monai"),
    ("SimpleITK",          "SimpleITK"),
    ("pandas",             "pandas"),
    ("tqdm",               "tqdm"),
    ("huggingface_hub",    "huggingface_hub"),
]

missing = []
for import_name, pkg_label in required:
    try:
        importlib.import_module(import_name)
    except Exception as exc:
        missing.append((import_name, pkg_label, exc))

if missing:
    print("Import check FAILED for:")
    for name, label, exc in missing:
        print(f"  - {label} ({name}): {type(exc).__name__}: {exc}")
    sys.exit(1)

import torch
print(f"torch       : {torch.__version__}")
print(f"cuda avail. : {torch.cuda.is_available()}  "
      f"({'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU detected'})")
print("Import check passed.")
PY

log "Done."
echo ""
echo "============================================================"
echo "RADAR runtime ready:"
echo "  interpreter  : $RADAR_PY"
echo "  repo         : $RADAR_REPO"
echo "  checkpoints  : $RADAR_REPO/ckpt"
echo "============================================================"