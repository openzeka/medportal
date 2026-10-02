# MedPortal

A single web interface over two medical imaging models, built to run on a single
NVIDIA DGX Spark:

- **RADAR** — expert-level abdominal CT analysis. You upload a contrast-enhanced
  abdominal CT (`.nii.gz`); it returns a probability score for each of 146 findings,
  plus an interactive slice viewer and CSV export.
- **ClinFusion-32B** — a vision-centric multimodal LLM. You ask questions in plain
  language and optionally attach 2D images (`.jpg`/`.png`) or 3D CT/MRI volumes
  (`.nii.gz`); it answers in free text and keeps the conversation context across turns.

MedPortal itself contains no model weights and does no inference of its own. It is
the orchestration layer: it serves the web UI, uploads files, serialises GPU access
behind a single queue, runs RADAR as a subprocess, and keeps the 32B model resident in
memory so chat requests do not pay a reload cost.

---

## The models

### RADAR — [github.com/alibaba-damo-academy/damo-radar](https://github.com/alibaba-damo-academy/damo-radar)

Published in *Science* (2026), RADAR is a generalist vision-language model trained on
more than 400,000 contrast-enhanced abdominal CT examinations with 15 million
anatomy-aware image–text pairs. It learns directly from clinical reports, without
manual annotation, and reports expert-level performance across routine and complex
clinical tasks.

Internally it pairs a 3D visual encoder (UNet-based, from nnU-Net) with a text encoder
(BERT) that embeds the 146 finding names; the image–text match produces one positive
probability per finding.

| | |
|---|---|
| **Upstream code** | [alibaba-damo-academy/damo-radar](https://github.com/alibaba-damo-academy/damo-radar) |
| **Checkpoints** | [huggingface.co/radar-generalist](https://huggingface.co/radar-generalist) (also on ModelScope) |
| **Paper** | [*Science* 393(6817), eaec6129](https://www.science.org/doi/10.1126/science.aec6129) |
| **License** | CC BY-NC-SA 4.0 — **research use only**, not cleared for clinical deployment |
| **Checkpoints size** | ~6 GB |

RADAR expects a contrast-enhanced abdominal CT with axial slices in HU. Inputs outside
that distribution (non-contrast CT, chest CT, other body regions) are out of its
training distribution and its scores should not be trusted.

### ClinFusion-32B — [github.com/alibaba-damo-academy/ClinFusion](https://github.com/alibaba-damo-academy/ClinFusion)

A vision-centric multimodal LLM system for holistic medical understanding
([arXiv:2607.24743](https://arxiv.org/abs/2607.24743)). Its base is **Qwen3-VL-32B-Instruct**,
enriched with additional visual encoders that give it native volumetric understanding:

| Encoder | Role |
|---|---|
| **DINOv2-large** | Self-supervised dense spatial semantics |
| **CLIP-ConvNeXt-large** | Structural and text-aligned appearance features |
| **3D positional encoding** | Depth/position information for volumetric inputs |

These are fused by a cascade spatial-aware locality fusion stage, so the model can
reason over a 3D CT volume rather than only over rendered 2D slices. An 8B variant
exists; this project uses the 32B one.

| | |
|---|---|
| **Upstream code** | [alibaba-damo-academy/ClinFusion](https://github.com/alibaba-damo-academy/ClinFusion) |
| **Checkpoints** | [huggingface.co/collections/Alibaba-DAMO-Academy/clinfusion](https://huggingface.co/collections/Alibaba-DAMO-Academy/clinfusion) |
| **Paper** | [arXiv:2607.24743](https://arxiv.org/abs/2607.24743) |
| **License** | Apache-2.0 |
| **Weights size** | ~180 GB (checkpoint + base LLM + both vision encoders) |

Upstream also ships a large evaluation suite (211K records over 22 medical benchmarks)
as a separate dataset release.

---

## Requirements

| | |
|---|---|
| **Hardware** | NVIDIA DGX Spark — GB10 GPU, 128 GB unified memory, aarch64 |
| **Storage** | ~250 GB free (≈6 GB RADAR + ≈180 GB ClinFusion + working space) |
| **OS** | Ubuntu 24.04 (tested on 24.04.4) |
| **Driver** | 580.173.02 or newer (CUDA 13.0) |
| **Package managers** | `git`, `conda`, `huggingface-cli`, and `uv` for the ClinFusion environment |

The 32B model is resident in memory once loaded. Budget roughly 75 GB of unified
memory while it is serving.

---

## Setup

### 1. Get the three codebases

```bash
git clone https://github.com/openzeka/medportal.git
git clone https://github.com/alibaba-damo-academy/damo-radar.git       ~/radar
git clone https://github.com/alibaba-damo-academy/ClinFusion.git      ~/ClinFusion
```

The two upstream repositories default to `~/radar` and `~/ClinFusion`, which is where
MedPortal looks for them. To keep them elsewhere, set `RADAR_REPO` and
`CLINFUSION_REPO` (see [Configuration](#configuration)).

### 2. RADAR — environment and checkpoints

```bash
conda create -n radar python=3.10 -y
conda activate radar
pip install -r ~/radar/requirements.txt

cd ~/radar/download_scripts
python download_checkpoints.py          # -> ~/radar/ckpt  (~6 GB)
```

### 3. ClinFusion — environment and checkpoints

ClinFusion's own `install_from_scratch.sh` does not work as-is for this stack — two
real issues surface on a fresh DGX Spark:

1. **Upstream's environment path is machine-specific.** It hardcodes
   `ENV_DIR=/tmp/hangjie.yhj/envs/qwen3-vl` (the upstream author's own path, inside
   `/tmp`, cleared on reboot), which does not match what MedPortal's backend uses:
   `$CONDA_ROOT/envs/clinfusion/bin/python`.
2. **The pinned FlashAttention wheel is for the wrong platform.**
   `requirements.txt` names `flash_attn-2.8.3+cu128torch2.8-cp311-cp311-linux_x86_64.whl`
   (x86_64 / CUDA 12.8 / torch 2.8), but DGX Spark is aarch64 / CUDA 13.0 / torch 2.14.

`scripts/setup_clinfusion.sh` fixes both: it installs the environment to the exact
path MedPortal expects, and pulls the matching prebuilt FlashAttention wheel
(aarch64 / CUDA 13.0 / torch 2.14 / CPython 3.11) from
[mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases).
It installs only the packages `custom_model/` actually imports — `vllm`, `ray`,
`deepspeed` etc. are used for ClinFusion's own evaluation harness and are not
needed by `worker/clinfusion_worker.py`, so they are left out.

```bash
cd ~/medportal
./scripts/setup_clinfusion.sh            # add --dry-run first to see resolved URLs
```

Then fetch the weights (~180 GB, expect this to take a while):

```bash
export HF_ENDPOINT=https://hf-mirror.com # optional mirror

huggingface-cli download --resume-download facebook/dinov2-large \
  --repo-type model --local-dir cache/models/dinov2-large

huggingface-cli download --resume-download \
  laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup \
  --repo-type model \
  --local-dir cache/models/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup

huggingface-cli download --resume-download Qwen/Qwen3-VL-32B-Instruct \
  --repo-type model --local-dir cache/models/Qwen3-VL-32B-Instruct

huggingface-cli download --resume-download Alibaba-DAMO-Academy/ClinFusion-32B \
  --repo-type model --local-dir cache/models/ClinFusion-32B
```

### 4. MedPortal — environment

```bash
conda create -n medportal python=3.11 -y
conda activate medportal
pip install -r medportal/backend/requirements.txt
```

Three environments are required and cannot be merged: RADAR needs `transformers 4.25`
(pinned for LAVIS compatibility) while ClinFusion needs `transformers 4.57.0`.

---

## Running

```bash
cd medportal
./run.sh
```

`run.sh` starts the ClinFusion worker first and waits for it to report ready. The 32B
model takes **about 14 minutes** to load on first start — progress is in
`logs/worker.log`:

```bash
tail -f logs/worker.log
```

Once the worker is ready the backend starts and the portal is reachable on port 8080:

```bash
http://<your-dgx-spark-address>:8080
```

| Command | What it does |
|---|---|
| `./run.sh` | Start the worker and the backend (waits for the model) |
| `./stop.sh` | Stop both, releasing GPU memory |
| `./stop.sh --force` | Same, but SIGKILL if a process lingers |
| `./status.sh` | Processes, ports, worker health, backend status |
| `./scripts/smoke.sh` | End-to-end check; expects `findings: 146` and `SMOKE OK` |

### Ports

| Port | Bound to | Purpose |
|---|---|---|
| 8080 | `0.0.0.0` | Web UI and HTTP API |
| 8100 | `127.0.0.1` | ClinFusion worker, internal only |

---

## Configuration

Every path and port can be overridden with environment variables, so the same checkout
runs on any machine:

| Variable | Default | Meaning |
|---|---|---|
| `RADAR_REPO` | `~/radar` | RADAR checkout |
| `RADAR_SCRIPT` | `$RADAR_REPO/RADAR_inference/inference_demo.py` | RADAR entry point |
| `RADAR_PY` | `$CONDA_ROOT/envs/radar/bin/python` | RADAR interpreter |
| `CLINFUSION_REPO` | `~/ClinFusion` | ClinFusion checkout |
| `CLINFUSION_MODEL` | `cache/models/ClinFusion-32B` | Checkpoint, relative to the repo |
| `CLINFUSION_PY` | `$CONDA_ROOT/envs/clinfusion/bin/python` | Worker interpreter |
| `MEDPORTAL_PY` | `$CONDA_ROOT/envs/medportal/bin/python` | Backend interpreter |
| `CONDA_ROOT` | `~/miniconda3` | Where your conda environments live |
| `MEDPORTAL_WORKSPACE` | `./workspace` | Uploads and job artefacts |
| `BACKEND_PORT` | `8080` | Web/API port |
| `WORKER_PORT` | `8100` | Worker port |
| `MAX_NEW_TOKENS` | `16384` | Answer length ceiling |
| `RADAR_TIMEOUT` | `600` | RADAR job timeout (seconds) |
| `CHAT_TIMEOUT` | `1800` | Chat generation timeout (seconds) |
| `MAX_UPLOAD_MB` | `500` | Upload size limit |

```bash
RADAR_REPO=/data/radar CLINFUSION_REPO=/data/ClinFusion ./run.sh
```

---

## Using it

### RADAR tab

Drop a `.nii.gz` file on the upload box. The slice viewer opens immediately and
analysis runs in the background; the findings table fills in when it completes.

- **Threshold** — shows only findings at or above the chosen probability. Presets are
  0.30 (sensitive), 0.50 (default) and 0.70 (selective). A lower threshold shows more
  findings and more noise.
- **Group by organ** — groups rows by organ, highest-scoring group first.
- **Windowing** — `Soft tissue`, `Lung`, `Bone` and `Full` presets, plus
  `←`/`→` for slices, `PgUp`/`PgDn` for ±10, `Home`/`End` for first/last, wheel to
  zoom and drag to pan.
- **Download CSV** — exports every score.

### ClinFusion tab

Attach images with `📎`, ask your question, press `Enter` (`Shift+Enter` for a newline).

- Attachments stay in the sidebar and are marked `✓ sent`; reuse them in the next
  question or remove them with `×`.
- A 3D CT attachment is evaluated as a set of representative slices plus the volume
  data. For a question about one specific region, cutting that region out as a 2D
  image and attaching it gives the more reliable answer.
- When an answer hits the length ceiling it is flagged rather than silently cut.

---

## Notes and limits

- **One job at a time.** RADAR and ClinFusion share the single GPU, so jobs are
  serialised through one queue. Submissions made while a job runs are queued; the
  status bar shows what is running and how many are waiting.
- **Scores are probabilities, not diagnoses.** Decisions stay with the clinician and
  the radiology report.
- **RADAR research use only** — its upstream license is non-commercial and explicitly
  not cleared for clinical deployment.
- **Timing.** Model load ~14 min; RADAR analyses ~10–30 s once loaded; chat answers
  range from seconds to several minutes depending on length.
- **RAM.** RADAR and the 32B worker cannot hold their peak simultaneously. MedPortal
  retries a RADAR job once if the model load has just released its peak.

---

## Development

```bash
cd backend && python -m pytest tests -q     # 74 tests
```

| Path | Contents |
|---|---|
| `backend/app/` | FastAPI backend, RADAR runner, NIfTI slicing |
| `backend/tests/` | Test suite |
| `worker/` | ClinFusion resident HTTP worker |
| `frontend/` | Single-page UI, no build step, no CDN dependencies |
| `scripts/smoke.sh` | End-to-end check |

---

## License and credits

MedPortal is the orchestration layer; the models carry their own licenses.

- **RADAR** — [CC BY-NC-SA 4.0](https://github.com/alibaba-damo-academy/damo-radar/blob/main/LICENSE), research use only
- **ClinFusion** — [Apache-2.0](https://github.com/alibaba-damo-academy/ClinFusion/blob/master/LICENSE)

If you use these models, cite the upstream work:

```bibtex
@article{damo-radar-2026,
    title   = {An expert-level generalist AI for abdominal CT diagnosis},
    journal = {Science},
    volume  = {393},
    number  = {6817},
    pages   = {eaec6129},
    year    = {2026},
    doi     = {10.1126/science.aec6129}
}

@article{yuan2026ClinFusion,
    title  = {ClinFusion: A Vision-Centric Multimodal LLM System for Holistic Medical Understanding},
    journal = {arXiv preprint arXiv:2607.24743},
    year   = {2026}
}
```