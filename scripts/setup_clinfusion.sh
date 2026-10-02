#!/usr/bin/env bash
# MedPortal — ClinFusion runtime environment setup for DGX Spark (Linux aarch64).
#
# Why this script exists instead of ClinFusion's own install_from_scratch.sh:
#
#  1. UPSTREAM PATH MISMATCH (problem #3)
#     install_from_scratch.sh hardcodes  ENV_DIR="/tmp/hangjie.yhj/envs/qwen3-vl" —
#     the upstream author's machine-specific path, inside /tmp (cleared on reboot),
#     which does not match what MedPortal's backend expects:
#         CLINFUSION_PY = $CONDA_ROOT/envs/clinfusion/bin/python
#     This script installs to that expected location (or an override via CONDA_ROOT /
#     CLINFUSION_PY), so no configuration change is needed afterwards.
#
#  2. WRONG FLASH-ATTENTION WHEEL (problem #4)
#     requirements.txt pins  flash_attn-2.8.3+cu128torch2.8-cp311-cp311-linux_x86_64.whl
#     — an x86_64 / CUDA 12.8 / torch 2.8 wheel. DGX Spark is aarch64 / CUDA 13.0 /
#     torch 2.14, so that filename does not resolve anywhere and the upstream
#     installer fails with "Local whl not found". This script downloads the matching
#     prebuilt wheel for the running platform from mjun0812's public release.
#
#  3. DONE, DOES NOT NEED vLLM / Ray / DeepSpeed
#     `worker/clinfusion_worker.py` drives the model directly through
#     custom_model.medevalkit_adapter_qwen3_vl. We list the packages that code
#     actually imports instead of pulling the full upstream requirements.txt, whose
#     training-path extras (vllm, ray, grpcio, protobuf, deepspeed) we do not need
#     and cannot reasonably install on a cluster without those services.
#
# Usage:
#   ./scripts/setup_clinfusion.sh            # detect everything and install
#   PYTHON_VERSION=3.11 ./scripts/setup_clinfusion.sh
#   CLINFUSION_PY=/somewhere/python ./scripts/setup_clinfusion.sh   # custom target
#   ./scripts/setup_clinfusion.sh --dry-run  # show resolved URLs / paths only
#
# Afterwards, ./run.sh (and backend config) picks up CLINFUSION_PY automatically.

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CLINFUSION_REPO="${CLINFUSION_REPO:-$HOME/ClinFusion}"
CLINFUSION_PY="${CLINFUSION_PY:-$CONDA_ROOT/envs/clinfusion/bin/python}"
CLINFUSION_CACHE="${CLINFUSION_CACHE:-$CLINFUSION_REPO/cache}"

# --- FlashAttention wheel selection (aarch64 / CUDA 13.0 / torch 2.14 / cp311) ----
# Every component is derivable from the running machine, but overridable so the
# same script works for future toolchain bumps without editing this file.
ARCH="$(uname -m)"               # aarch64 | x86_64
TORCH_TAG="${TORCH_TAG:-2.14}"   # matched against the wheel URL
CUDA_TAG="${CUDA_TAG:-cu130}"    # expected by the torch build shipped with the wheel
WHL_PYTAG="${WHL_PYTAG:-cp311}"  # CPython 3.11
FA_BASE_URL="${FA_BASE_URL:-https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download}"
FA_RELEASE="${FA_RELEASE:-v0.10.1}"
FA_WHEEL_NAME="flash_attn-2.8.3+${CUDA_TAG}torch${TORCH_TAG}-${WHL_PYTAG}-${WHL_PYTAG}-linux_${ARCH}.whl"
FA_WHEEL_URL="${FA_BASE_URL}/${FA_RELEASE}/${FA_WHEEL_NAME}"

log()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# --- Sanity checks ---------------------------------------------------------------
command -v curl >/dev/null 2>&1 || die "curl is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"

if [[ "$ARCH" != "aarch64" ]]; then
    warn "uname -m is '$ARCH', not aarch64. Wheel selection adapts, but this path"
    warn "has only been tested on DGX Spark (Linux aarch64)."
fi

if [[ ! -d "$CLINFUSION_REPO" ]]; then
    die "ClinFusion repo not found at $CLINFUSION_REPO. Clone it first, or set CLINFUSION_REPO."
fi
[[ -d "$CLINFUSION_REPO/custom_model" ]] \
    || die "$CLINFUSION_REPO does not look like a ClinFusion checkout (custom_model/ missing)"

log "Platform           : $ARCH"
log "Target interpreter  : $CLINFUSION_PY"
log "FlashAttention wheel: $FA_WHEEL_NAME"
log "Wheel URL           : $FA_WHEEL_URL"

if [[ "$DRY_RUN" == "1" ]]; then
    log "--dry-run: nothing else was done."
    exit 0
fi

# --- Wheel availability check ----------------------------------------------------
# GitHub release downloads always respond 302 first (redirect to the CDN);
# follow the redirect chain before checking the final status code.
HTTP_CODE="$(curl -sIL -o /dev/null -w '%{http_code}' "$FA_WHEEL_URL" || echo 0)"
[[ "$HTTP_CODE" == "200" ]] \
    || die "FlashAttention wheel not found for this platform/tag (HTTP $HTTP_CODE): $FA_WHEEL_URL
Check TORCH_TAG/CUDA_TAG/WHL_PYTAG/FA_RELEASE, or search
https://github.com/mjun0812/flash-attention-prebuild-wheels/releases for a matching wheel."

# --- Environment creation --------------------------------------------------------
ENV_DIR="$(dirname "$(dirname "$CLINFUSION_PY")")"   # .../envs/clinfusion
mkdir -p "$ENV_DIR"

if [[ -x "$CLINFUSION_PY" ]]; then
    log "Interpreter already exists at $CLINFUSION_PY, reusing it."
elif command -v uv >/dev/null 2>&1; then
    log "Creating environment at $ENV_DIR with uv (Python $PYTHON_VERSION)..."
    uv venv "$ENV_DIR" --python "$PYTHON_VERSION"
elif [[ -x "$CONDA_ROOT/bin/conda" ]]; then
    log "Creating conda environment at $ENV_DIR..."
    "$CONDA_ROOT/bin/conda" create -y -p "$ENV_DIR" "python=$PYTHON_VERSION"
else
    die "Neither uv nor conda is available. Install one of them and re-run."
fi

[[ -x "$CLINFUSION_PY" ]] || die "Interpreter not created as expected: $CLINFUSION_PY"

# --- Fetch the platform-correct FlashAttention wheel ----------------------------
WHEEL_CACHE="${WHEEL_CACHE:-$CLINFUSION_CACHE/wheels}"
mkdir -p "$WHEEL_CACHE"
WHEEL_PATH="$WHEEL_CACHE/$FA_WHEEL_NAME"

if [[ -f "$WHEEL_PATH" ]]; then
    log "FlashAttention wheel already cached at $WHEEL_PATH, skipping download."
else
    log "Downloading FlashAttention wheel..."
    curl -fL --retry 3 --retry-delay 2 -o "$WHEEL_PATH.part" "$FA_WHEEL_URL"
    mv "$WHEEL_PATH.part" "$WHEEL_PATH"
fi

# --- Runtime dependencies ----------------------------------------------------
# Verified by inspecting two sources: (1) ClinFusion/custom_model/**.py imports
# (the model side), and (2) medportal's worker/clinfusion_worker.py imports (the
# service side) — this second group is easy to forget (fastapi/uvicorn/pydantic
# live here, not in the model code) and is where a fresh install previously broke
# with "ModuleNotFoundError: No module named 'uvicorn'". Nothing on this list is
# speculative.
log "Installing ClinFusion runtime dependencies..."
"$CLINFUSION_PY" -m pip install --upgrade pip
"$CLINFUSION_PY" -m pip install \
    "torch" "torchvision" \
    "transformers==4.57.0" \
    einops \
    open_clip_torch \
    timm \
    monai \
    nibabel \
    scipy \
    matplotlib \
    packaging \
    pillow \
    accelerate \
    huggingface_hub \
    fastapi \
    uvicorn \
    pydantic \
    deepspeed

# FlashAttention last — it must compile/link against the torch build above.
log "Installing FlashAttention (prebuilt)..."
"$CLINFUSION_PY" -m pip install "$WHEEL_PATH"

# --- Smoke check: can we import what the worker will need? ----------------------
log "Verifying imports..."
"$CLINFUSION_PY" - <<'PY'
import importlib, sys

required = [
    ("torch",                  "torch"),
    ("torchvision",            "torchvision"),
    ("transformers",           "transformers"),
    ("open_clip",              "open_clip"),
    ("timm",                   "timm"),
    ("einops",                 "einops"),
    ("monai",                  "monai"),
    ("nibabel",                "nibabel"),
    ("PIL",                    "Pillow"),
    ("flash_attn",             "flash-attn"),
]

missing = []
for import_name, pkg_label in required:
    try:
        importlib.import_module(import_name)
    except Exception as exc:  # noqa: BLE001 - report and continue
        missing.append((import_name, pkg_label, exc))

if missing:
    print("Import check FAILED for:")
    for name, label, exc in missing:
        print(f"  - {label} ({name}): {type(exc).__name__}: {exc}")
    sys.exit(1)

import torch, flash_attn
print(f"torch        : {torch.__version__}")
print(f"cuda avail.  : {torch.cuda.is_available()}  "
      f"({'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU detected'})")
print(f"flash_attn   : {flash_attn.__version__}")
print("Import check passed.")
PY

log "Done."
echo ""
echo "============================================================"
echo "ClinFusion runtime ready:"
echo "  interpreter : $CLINFUSION_PY"
echo "  repo        : $CLINFUSION_REPO"
echo "  model cache : $CLINFUSION_CACHE/models  (download separately if missing)"
echo "============================================================"