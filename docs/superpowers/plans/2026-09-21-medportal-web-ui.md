# MedPortal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** RADAR ve ClinFusion-32B'yi tek bir web portalında sunmak (RADAR: BT → kesit görüntüleyici + 146 bulgu tablosu; ClinFusion: çok turlu sohbet).

**Architecture:** FastAPI tabanlı orkestratör backend (`medportal` env), iki farklı conda env'deki modelleri yönetir: RADAR istek başına subprocess (env `radar`), ClinFusion-32B sürekli açık localhost HTTP worker (env `clinfusion`). Tek GPU için asyncio kuyruğu. Ön uç Node gerektirmeyen tek sayfa (vanilla JS + Tailwind CDN).

**Tech Stack:** Python 3.11 (FastAPI, uvicorn, httpx, nibabel, Pillow), Vanilla JS + Tailwind CDN, conda env'leri `medportal`/`radar`/`clinfusion`, DGX Spark aarch64.

**Referans spec:** `docs/superpowers/specs/2026-09-21-medportal-web-ui-design.md`

**Çalışma dizini (tüm yollar buna göre):** `~/medportal` (Spark: 192.168.1.162, kullanıcı `nvidia`)

**Ortam yolları:**
- Backend python: `/home/nvidia/miniconda3/envs/medportal/bin/python`
- RADAR python: `/home/nvidia/miniconda3/envs/radar/bin/python`
- ClinFusion python: `/home/nvidia/miniconda3/envs/clinfusion/bin/python`
- RADAR script: `/home/nvidia/radar/RADAR_inference/inference_demo.py`
- ClinFusion repo: `/home/nvidia/ClinFusion`

---

## Dosya Yapısı

```
~/medportal/
├── README.md                                   (mevcut)
├── .gitignore                                  (mevcut)
├── run.sh                                      (oluştur)
├── backend/
│   ├── requirements.txt                        (oluştur)
│   ├── pytest.ini                              (oluştur)
│   ├── app/
│   │   ├── __init__.py                         (oluştur)
│   │   ├── config.py                           (oluştur) yollar/limitler
│   │   ├── schemas.py                          (oluştur) pydantic modelleri
│   │   ├── store.py                            (oluştur) job store (in-memory)
│   │   ├── gpu_queue.py                        (oluştur) tek-GPU asyncio kilidi
│   │   ├── nifti.py                            (oluştur) kesit PNG üretimi
│   │   ├── radar.py                            (oluştur) RADAR subprocess + CSV parse
│   │   ├── clinfusion.py                       (oluştur) worker HTTP istemcisi
│   │   ├── uploads.py                          (oluştur) yükleme doğrulama/kayıt
│   │   └── main.py                             (oluştur) FastAPI app + rotalar
│   └── tests/
│       ├── conftest.py                         (oluştur)
│       ├── test_config.py                      (oluştur)
│       ├── test_store.py                       (oluştur)
│       ├── test_queue.py                       (oluştur)
│       ├── test_nifti.py                       (oluştur)
│       ├── test_uploads.py                     (oluştur)
│       ├── test_radar.py                       (oluştur)
│       ├── test_clinfusion.py                  (oluştur)
│       ├── test_worker_contract.py             (oluştur)
│       └── test_api.py                         (oluştur)
├── worker/
│   ├── __init__.py                             (oluştur)
│   └── clinfusion_worker.py                    (oluştur) 32B resident HTTP worker
├── frontend/
│   ├── index.html                              (oluştur)
│   ├── styles.css                              (oluştur)
│   └── app.js                                  (oluştur)
└── scripts/
    └── smoke.sh                                (oluştur) entegrasyon duman testi
```

**Sorumluluk sınırları:**
- `config.py`: tüm sabit yollar/limitler tek yerde.
- `store.py`: job durum kaydı (queued/running/done/error) — GPU/orkestrasyondan bağımsız.
- `gpu_queue.py`: yalnız GPU'ya erişen işleri sıraya sokar.
- `radar.py`: RADAR CLI'ını çağırır ve CSV'yi `findings[]`'e çevirir (HTTP bilmez).
- `clinfusion.py`: yalnız worker'a HTTP konuşur.
- `uploads.py`: dosya doğrulama + diske yazma.
- `main.py`: rotalar; iş mantığını üstteki modüllere delege eder.
- `worker/clinfusion_worker.py`: model yükleme + üretim; backend'den bağımsız çalışabilir.

---

## Task 0: `medportal` env + iskelet

**Files:**
- Create: `backend/requirements.txt`, `backend/pytest.ini`, `backend/app/__init__.py`, `worker/__init__.py`

- [ ] **Step 1: Env oluştur (Spark'te, ssh ile)**

Run:
```bash
~/miniconda3/bin/conda create -n medportal python=3.11 -y
```

- [ ] **Step 2: `backend/requirements.txt` yaz**

```
fastapi==0.115.*
uvicorn[standard]==0.34.*
python-multipart==0.0.20
httpx==0.28.*
nibabel==5.4.*
numpy==2.2.*
pillow==12.*
pytest==8.3.*
pytest-asyncio==1.0.*
```

- [ ] **Step 3: `backend/pytest.ini` yaz**

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 4: Boş paket dosyaları oluştur**

`backend/app/__init__.py` ve `worker/__init__.py` boş içerik.

- [ ] **Step 5: Bağımlılıkları kur**

Run:
```bash
/home/nvidia/miniconda3/envs/medportal/bin/pip install -r backend/requirements.txt
```
Expected: "Successfully installed ..."

- [ ] **Step 6: Commit**

```bash
git add backend/requirements.txt backend/pytest.ini backend/app/__init__.py worker/__init__.py
git commit -m "chore: medportal env scaffold and backend deps"
```

---

## Task 1: `config.py` — yollar ve limitler

**Files:**
- Create: `backend/app/config.py`
- Test: `backend/tests/conftest.py` (yardımcılar burada)

- [ ] **Step 1: Failing test yaz** — `backend/tests/test_config.py`

```python
from app import config

def test_paths_are_absolute():
    assert config.ROOT.is_absolute()
    assert config.UPLOADS.is_absolute()
    assert config.JOBS.is_absolute()
    assert config.FRONTEND.is_absolute()

def test_defaults():
    assert config.MAX_UPLOAD_MB == 500
    assert config.RADAR_TIMEOUT == 600
    assert config.CHAT_TIMEOUT == 1800
    assert config.WORKER_URL.startswith("http://127.0.0.1")
```

- [ ] **Step 2: Çalıştır, fail gör**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/test_config.py -v`
Expected: FAIL (ModuleNotFoundError: app.config)

- [ ] **Step 3: `backend/app/config.py` yaz**

```python
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
```

- [ ] **Step 4: `backend/tests/conftest.py` yaz**

```python
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
```

- [ ] **Step 5: Testleri geçir**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add backend/app/config.py backend/tests/conftest.py backend/tests/test_config.py
git commit -m "feat: medportal config"
```

---

## Task 2: `store.py` + `gpu_queue.py` — job kaydı ve tek-GPU kilidi

**Files:**
- Create: `backend/app/store.py`, `backend/app/gpu_queue.py`, `backend/app/schemas.py`
- Test: `backend/tests/test_store.py`, `backend/tests/test_queue.py`

- [ ] **Step 1: Failing testler** — `backend/tests/test_store.py`

```python
from app.store import JobStore

def test_create_and_update():
    s = JobStore()
    jid = s.create("radar")
    assert s.get(jid)["status"] == "queued"
    s.set_running(jid)
    assert s.get(jid)["status"] == "running"
    s.set_done(jid, {"findings": []})
    assert s.get(jid)["status"] == "done"
    assert s.get(jid)["result"] == {"findings": []}

def test_error():
    s = JobStore()
    jid = s.create("chat")
    s.set_error(jid, "boom")
    assert s.get(jid)["status"] == "error"
    assert "boom" in s.get(jid)["error"]
```

`backend/tests/test_queue.py`

```python
import asyncio
from app.gpu_queue import GpuQueue

async def test_serialized():
    q = GpuQueue()
    order = []

    async def job(n, d):
        async with q.lock("job%d" % n):
            order.append(("start", n))
            await asyncio.sleep(d)
            order.append(("end", n))

    await asyncio.gather(job(1, 0.05), job(2, 0.01))
    assert order == [("start", 1), ("end", 1), ("start", 2), ("end", 2)]

def test_current_job():
    q = GpuQueue()
    assert q.current_job is None
    q.current_job = "abc"
    assert q.current_job == "abc"
```

- [ ] **Step 2: Fail gör**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/test_store.py backend/tests/test_queue.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: `backend/app/store.py` yaz**

```python
import threading
import uuid
import time
from typing import Any, Dict, Optional


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create(self, kind: str) -> str:
        jid = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[jid] = {
                "id": jid,
                "kind": kind,
                "status": "queued",
                "result": None,
                "error": None,
                "created": time.time(),
            }
        return jid

    def _update(self, jid: str, **kw: Any) -> None:
        with self._lock:
            if jid in self._jobs:
                self._jobs[jid].update(kw)

    def set_running(self, jid: str) -> None:
        self._update(jid, status="running")

    def set_done(self, jid: str, result: Any) -> None:
        self._update(jid, status="done", result=result)

    def set_error(self, jid: str, error: str) -> None:
        self._update(jid, status="error", error=error)

    def get(self, jid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(jid)
            return dict(job) if job else None


STORE = JobStore()
```

- [ ] **Step 4: `backend/app/gpu_queue.py` yaz**

```python
import asyncio
from contextlib import asynccontextmanager
from typing import Optional


class GpuQueue:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.current_job: Optional[str] = None
        self.pending: int = 0

    @asynccontextmanager
    async def lock(self, job_id: str):
        self.pending += 1
        try:
            async with self._lock:
                self.current_job = job_id
                try:
                    yield
                finally:
                    self.current_job = None
        finally:
            self.pending -= 1


QUEUE = GpuQueue()
```

- [ ] **Step 5: `backend/app/schemas.py` yaz**

```python
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    upload_id: str
    kind: str
    filename: str
    n_slices: Optional[int] = None


class RadarRequest(BaseModel):
    upload_id: str


class JobIdResponse(BaseModel):
    job_id: str


class HistoryTurn(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    prompt: str
    history: List[HistoryTurn] = []
    attachment_ids: List[str] = []


class StatusResponse(BaseModel):
    gpu: str = "GB10"
    radar_ready: bool = True
    clinfusion_ready: bool
    busy: bool
    current_job: Optional[str]
    queue_len: int


class JobResponse(BaseModel):
    id: str
    kind: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
```

- [ ] **Step 6: Testleri geçir**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/test_store.py backend/tests/test_queue.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit**

```bash
git add backend/app/store.py backend/app/gpu_queue.py backend/app/schemas.py backend/tests/test_store.py backend/tests/test_queue.py
git commit -m "feat: job store and single-GPU queue"
```

---

## Task 3: `nifti.py` — aksiyel kesit PNG

**Files:**
- Create: `backend/app/nifti.py`
- Test: `backend/tests/test_nifti.py`

- [ ] **Step 1: Failing test** — `backend/tests/test_nifti.py`

```python
import numpy as np
import nibabel as nib
from app.nifti import load_volume, slice_png

def _make_nii(tmp_path):
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    img = nib.Nifti1Image(data, np.eye(4))
    p = tmp_path / "v.nii.gz"
    nib.save(img, str(p))
    return p

def test_load_volume(tmp_path):
    p = _make_nii(tmp_path)
    vol = load_volume(p)
    assert vol.shape == (4, 5, 6)

def test_n_slices(tmp_path):
    p = _make_nii(tmp_path)
    assert load_volume(p).shape[2] == 6

def test_slice_png(tmp_path):
    p = _make_nii(tmp_path)
    png = slice_png(p, 3)
    assert png[:4] == b"\x89PNG"
```

- [ ] **Step 2: Fail gör**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/test_nifti.py -v`
Expected: FAIL

- [ ] **Step 3: `backend/app/nifti.py` yaz**

```python
import io
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

WINDOW_MIN = -1000.0
WINDOW_MAX = 1000.0


def load_volume(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data


def n_slices(path: Path) -> int:
    return int(load_volume(path).shape[2])


def _to_uint8(sl: np.ndarray) -> np.ndarray:
    sl = np.clip(sl, WINDOW_MIN, WINDOW_MAX)
    sl = (sl - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    return (sl * 255.0).astype(np.uint8)


def slice_png(path: Path, idx: int) -> bytes:
    vol = load_volume(path)
    idx = max(0, min(idx, vol.shape[2] - 1))
    # aksiyel dilim (z ekseni); görüntüyü dik tutmak için transpose
    sl = np.rot90(vol[:, :, idx])
    arr = _to_uint8(sl)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()
```

- [ ] **Step 4: Testleri geçir**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/test_nifti.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/app/nifti.py backend/tests/test_nifti.py
git commit -m "feat: nifti axial slice rendering"
```

---

## Task 4: `uploads.py` — yükleme doğrulama

**Files:**
- Create: `backend/app/uploads.py`
- Test: `backend/tests/test_uploads.py`

- [ ] **Step 1: Failing test** — `backend/tests/test_uploads.py`

```python
import pytest
from app import config
from app.uploads import save_upload, UploadError

@pytest.fixture(autouse=True)
def _dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOADS", tmp_path / "uploads")
    config.UPLOADS.mkdir(parents=True, exist_ok=True)

def test_save_nifti():
    meta = save_upload("scan.nii.gz", b"x" * 100)
    assert meta["kind"] == "nifti"
    assert (config.UPLOADS / meta["upload_id"] / "scan.nii.gz").is_file()

def test_reject_bad_ext():
    with pytest.raises(UploadError):
        save_upload("evil.exe", b"x")

def test_reject_too_big(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)
    with pytest.raises(UploadError):
        save_upload("scan.nii.gz", b"x" * 10)
```

- [ ] **Step 2: Fail gör**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/test_uploads.py -v`
Expected: FAIL

- [ ] **Step 3: `backend/app/uploads.py` yaz**

```python
import uuid
from pathlib import Path

from . import config


class UploadError(Exception):
    pass


def _kind(name: str) -> str:
    lower = name.lower()
    if lower.endswith(config.ALLOWED_NIFTI):
        return "nifti"
    if lower.endswith(config.ALLOWED_IMAGE):
        return "image"
    raise UploadError(f"Desteklenmeyen dosya türü: {name}")


def save_upload(filename: str, data: bytes) -> dict:
    kind = _kind(filename)
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise UploadError("Dosya çok büyük")
    uid = uuid.uuid4().hex[:12]
    dest_dir = config.UPLOADS / uid
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = Path(filename).name
    (dest_dir / safe).write_bytes(data)
    return {"upload_id": uid, "kind": kind, "filename": safe, "path": dest_dir / safe}


def resolve(upload_id: str) -> Path:
    d = config.UPLOADS / upload_id
    if not d.is_dir():
        raise UploadError("upload_id bulunamadı")
    files = [p for p in d.iterdir() if p.is_file()]
    if not files:
        raise UploadError("yükleme boş")
    return files[0]
```

- [ ] **Step 4: Testleri geçir**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/test_uploads.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/app/uploads.py backend/tests/test_uploads.py
git commit -m "feat: upload validation and storage"
```

---

## Task 5: `radar.py` — subprocess + CSV parse

**Files:**
- Create: `backend/app/radar.py`
- Test: `backend/tests/test_radar.py`

- [ ] **Step 1: Failing test** — `backend/tests/test_radar.py`

```python
import csv
from pathlib import Path
from app.radar import parse_csv, build_command

def test_parse_csv(tmp_path):
    p = tmp_path / "RADAR_infer_results_demo.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "肝_硬化 (Liver_Cirrhosis)", "肺_结节 (Lung_Nodule)"])
        w.writerow(["x.nii.gz", "0.9", "0.1"])
    findings = parse_csv(p)
    assert findings[0]["name"] == "Liver_Cirrhosis"
    assert findings[0]["score"] == 0.9
    assert findings[1]["name"] == "Lung_Nodule"

def test_build_command(tmp_path):
    cmd = build_command("radar", tmp_path / "in", tmp_path / "out")
    assert cmd[0].endswith("/envs/radar/bin/python")
    assert "--img_dir" in cmd and "--save_tag" in cmd
```

- [ ] **Step 2: Fail gör**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/test_radar.py -v`
Expected: FAIL

- [ ] **Step 3: `backend/app/radar.py` yaz**

```python
import asyncio
import csv
import re
import shutil
from pathlib import Path

from . import config


def build_command(tag: str, in_dir: Path, out_dir: Path):
    return [
        config.RADAR_PY,
        config.RADAR_SCRIPT,
        "--img_dir", str(in_dir),
        "--save_dir", str(out_dir),
        "--save_tag", tag,
    ]


_EN = re.compile(r"\(([^)]+)\)\s*$")


def parse_csv(path: Path):
    findings = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    for label, value in zip(header[1:], row[1:]):
        m = _EN.search(label)
        name = m.group(1) if m else label
        findings.append({"name": name, "label": label, "score": float(value)})
    findings.sort(key=lambda x: x["score"], reverse=True)
    return findings


async def run_radar(nifti_path: Path, job_dir: Path, job_id: str):
    in_dir = job_dir / "in"
    out_dir = job_dir / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(nifti_path, in_dir / nifti_path.name)
    log_path = job_dir / "job.log"

    cmd = build_command(job_id, in_dir, out_dir)
    with open(log_path, "w") as logf:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=logf,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(Path(config.RADAR_SCRIPT).parent),
        )
        try:
            code = await asyncio.wait_for(proc.wait(), config.RADAR_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError("RADAR zaman aşımı")

    if code != 0:
        tail = log_path.read_text(errors="replace")[-2000:]
        raise RuntimeError(f"RADAR başarısız (exit {code}):\n{tail}")

    csv_path = out_dir / f"RADAR_infer_results_{job_id}.csv"
    if not csv_path.is_file():
        raise RuntimeError("RADAR CSV üretmedi")
    findings = parse_csv(csv_path)
    return {"findings": findings, "csv_name": csv_path.name}
```

- [ ] **Step 4: Testleri geçir**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/test_radar.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/app/radar.py backend/tests/test_radar.py
git commit -m "feat: radar subprocess runner and csv parser"
```

---

## Task 6: `clinfusion.py` — worker istemcisi

**Files:**
- Create: `backend/app/clinfusion.py`
- Test: `backend/tests/test_clinfusion.py`

- [ ] **Step 1: Failing test** — `backend/tests/test_clinfusion.py`

```python
import httpx
import pytest
from app import clinfusion

async def test_generate_ok(monkeypatch):
    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"reply": "merhaba"}
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, json): 
            assert json["prompt"] == "selam"
            return FakeResp()
    monkeypatch.setattr(clinfusion.httpx, "AsyncClient", lambda **k: FakeClient())
    out = await clinfusion.generate(history=[], prompt="selam", attachment_paths=[])
    assert out == "merhaba"

async def test_generate_timeout(monkeypatch):
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, json): raise httpx.TimeoutException("boom")
    monkeypatch.setattr(clinfusion.httpx, "AsyncClient", lambda **k: FakeClient())
    with pytest.raises(clinfusion.ClinFusionError):
        await clinfusion.generate(history=[], prompt="x", attachment_paths=[])

async def test_health(monkeypatch):
    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"ready": True, "loading": False}
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, url): return FakeResp()
    monkeypatch.setattr(clinfusion.httpx, "AsyncClient", lambda **k: FakeClient())
    assert (await clinfusion.health())["ready"] is True
```

- [ ] **Step 2: Fail gör**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/test_clinfusion.py -v`
Expected: FAIL

- [ ] **Step 3: `backend/app/clinfusion.py` yaz**

```python
import httpx

from . import config


class ClinFusionError(Exception):
    pass


async def health() -> dict:
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{config.WORKER_URL}/health")
        r.raise_for_status()
        return r.json()


async def generate(history, prompt: str, attachment_paths) -> str:
    payload = {
        "history": [{"role": t.role if hasattr(t, "role") else t["role"],
                     "text": t.text if hasattr(t, "text") else t["text"]} for t in history],
        "prompt": prompt,
        "attachment_paths": [str(p) for p in attachment_paths],
    }
    try:
        async with httpx.AsyncClient(timeout=config.CHAT_TIMEOUT) as client:
            r = await client.post(f"{config.WORKER_URL}/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            if "reply" not in data:
                raise ClinFusionError(data.get("error", "worker cevabı geçersiz"))
            return data["reply"]
    except httpx.TimeoutException as e:
        raise ClinFusionError(f"ClinFusion zaman aşımı: {e}") from e
    except httpx.HTTPError as e:
        raise ClinFusionError(f"ClinFusion hatası: {e}") from e
```

- [ ] **Step 4: Testleri geçir**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/test_clinfusion.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/app/clinfusion.py backend/tests/test_clinfusion.py
git commit -m "feat: clinfusion worker client"
```

---

## Task 7: `main.py` — FastAPI rotaları

**Files:**
- Create: `backend/app/main.py`
- Test: `backend/tests/test_api.py`
- Modify: `backend/requirements.txt` ( `pytest` ile `httpx` zaten var; ek yok)

- [ ] **Step 1: Failing testler** — `backend/tests/test_api.py`

```python
import io
import numpy as np
import nibabel as nib
import pytest
from fastapi.testclient import TestClient

from app import config, store as store_mod
from app.main import app

@pytest.fixture(autouse=True)
def _dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(config, "JOBS", tmp_path / "jobs")
    config.UPLOADS.mkdir(parents=True, exist_ok=True)
    config.JOBS.mkdir(parents=True, exist_ok=True)
    store_mod.STORE._jobs.clear()
    yield

client = TestClient(app)

def _nii_bytes():
    data = np.zeros((4, 5, 6), dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    buf = io.BytesIO()
    nib.save(img, buf)
    return buf.getvalue()

def test_status():
    r = client.get("/api/status")
    assert r.status_code == 200
    assert "clinfusion_ready" in r.json()

def test_upload_and_slice():
    r = client.post("/api/upload", files={"file": ("v.nii.gz", _nii_bytes())})
    assert r.status_code == 200
    uid = r.json()["upload_id"]
    assert r.json()["n_slices"] == 6
    r2 = client.get(f"/api/nifti/{uid}/slice?idx=2")
    assert r2.status_code == 200
    assert r2.headers["X-Total-Slices"] == "6"

def test_radar_job(monkeypatch):
    async def fake_run(nifti_path, job_dir, job_id):
        return {"findings": [{"name": "X", "label": "X", "score": 0.9}], "csv_name": "c.csv"}
    import app.radar as radar
    monkeypatch.setattr(radar, "run_radar", fake_run)
    import app.main as main
    monkeypatch.setattr(main, "run_radar", fake_run)

    up = client.post("/api/upload", files={"file": ("v.nii.gz", _nii_bytes())}).json()
    jid = client.post("/api/radar", json={"upload_id": up["upload_id"]}).json()["job_id"]
    r = client.get(f"/api/radar/{jid}")
    assert r.status_code == 200
    assert r.json()["status"] in ("queued", "running", "done")

def test_chat_job(monkeypatch):
    import app.main as main
    async def fake_generate(history, prompt, attachment_paths):
        return "cevap"
    monkeypatch.setattr(main.cf, "generate", fake_generate)
    jid = client.post("/api/chat", json={"prompt": "selam"}).json()["job_id"]
    r = client.get(f"/api/chat/{jid}")
    assert r.status_code == 200
```

- [ ] **Step 2: Fail gör**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/test_api.py -v`
Expected: FAIL

- [ ] **Step 3: `backend/app/main.py` yaz**

```python
import asyncio
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import clinfusion as cf
from . import config, nifti
from .gpu_queue import QUEUE
from .radar import run_radar
from .schemas import ChatRequest, JobIdResponse, RadarRequest, StatusResponse, UploadResponse
from .store import STORE
from .uploads import UploadError, resolve, save_upload

app = FastAPI(title="MedPortal")


@app.on_event("startup")
async def _startup():
    config.ensure_dirs()


async def _radar_task(job_id: str, nifti_path: Path):
    async with QUEUE.lock(job_id):
        STORE.set_running(job_id)
        try:
            job_dir = config.JOBS / job_id
            result = await run_radar(nifti_path, job_dir, job_id)
            STORE.set_done(job_id, result)
        except Exception as e:
            STORE.set_error(job_id, str(e))


async def _chat_task(job_id: str, history, prompt, attachments):
    async with QUEUE.lock(job_id):
        STORE.set_running(job_id)
        try:
            reply = await cf.generate(history, prompt, attachments)
            STORE.set_done(job_id, {"reply": reply})
        except Exception as e:
            STORE.set_error(job_id, str(e))


@app.get("/api/status", response_model=StatusResponse)
async def status():
    try:
        h = await cf.health()
        ready = bool(h.get("ready"))
    except Exception:
        ready = False
    return StatusResponse(
        clinfusion_ready=ready,
        busy=QUEUE.current_job is not None,
        current_job=QUEUE.current_job,
        queue_len=QUEUE.pending,
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    try:
        meta = save_upload(file.filename, data)
    except UploadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    n = None
    if meta["kind"] == "nifti":
        n = nifti.n_slices(meta["path"])
    return UploadResponse(upload_id=meta["upload_id"], kind=meta["kind"],
                          filename=meta["filename"], n_slices=n)


@app.get("/api/nifti/{upload_id}/slice")
async def slice_(upload_id: str, idx: int = 0):
    try:
        path = resolve(upload_id)
    except UploadError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not str(path).lower().endswith(config.ALLOWED_NIFTI):
        raise HTTPException(status_code=400, detail="NIfTI değil")
    total = nifti.n_slices(path)
    png = nifti.slice_png(path, idx)
    return Response(content=png, media_type="image/png",
                    headers={"X-Total-Slices": str(total)})


@app.post("/api/radar", response_model=JobIdResponse)
async def radar_start(req: RadarRequest):
    try:
        path = resolve(req.upload_id)
    except UploadError as e:
        raise HTTPException(status_code=404, detail=str(e))
    jid = STORE.create("radar")
    asyncio.create_task(_radar_task(jid, path))
    return JobIdResponse(job_id=jid)


@app.get("/api/radar/{job_id}")
async def radar_get(job_id: str):
    job = STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job yok")
    result = job.get("result") or {}
    return {
        "status": job["status"],
        "progress": None,
        "findings": result.get("findings"),
        "csv_url": f"/api/radar/{job_id}/csv" if job["status"] == "done" else None,
        "error": job["error"],
    }


@app.get("/api/radar/{job_id}/csv")
async def radar_csv(job_id: str):
    csv_path = config.JOBS / job_id / "out" / f"RADAR_infer_results_{job_id}.csv"
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail="csv yok")
    return FileResponse(csv_path, filename=csv_path.name, media_type="text/csv")


@app.post("/api/chat", response_model=JobIdResponse)
async def chat_start(req: ChatRequest):
    attachments = []
    for uid in req.attachment_ids:
        try:
            attachments.append(resolve(uid))
        except UploadError as e:
            raise HTTPException(status_code=404, detail=str(e))
    jid = STORE.create("chat")
    asyncio.create_task(_chat_task(jid, req.history, req.prompt, attachments))
    return JobIdResponse(job_id=jid)


@app.get("/api/chat/{job_id}")
async def chat_get(job_id: str):
    job = STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job yok")
    result = job.get("result") or {}
    return {"status": job["status"], "reply": result.get("reply"), "error": job["error"]}


@app.post("/api/chat/reset")
async def chat_reset():
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(config.FRONTEND), html=True, check_dir=False), name="static")
```

- [ ] **Step 4: Testleri geçir**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/ -v`
Expected: PASS (tüm testler)

- [ ] **Step 5: Commit**

```bash
git add backend/app/main.py backend/tests/test_api.py
git commit -m "feat: fastapi routes for upload, radar, chat, nifti slices"
```

---

## Task 8: `worker/clinfusion_worker.py` — 32B resident worker

**Files:**
- Create: `worker/clinfusion_worker.py`
- Test: `backend/tests/test_worker_contract.py` (worker mantığını adapter'sız doğrulamak için izole fonksiyon)

> **Adapter API doğrulaması (bu göreve başlamadan önce):** Worker, `MedEvalKitAdapter`'in şu üyelerini kullanır: `_pre_sample_slices_from_volume(inputs, processor) -> (volume, processed)`, `_read_pil_image`, `llm`, `processor`, `device`, `max_new_tokens`, `eos_token_id`, `temperature`, `top_p`. Bunlar `~/ClinFusion/custom_model/medevalkit_adapter_qwen3_vl.py` içinde mevcuttur (satır ~267-363, ~473-513). Uygulama sırasında bu dosyayı okuyup imzaları teyit et; özellikle `processor(..., images_kwargs={"volume": [...]})` çağrısının 3D yolunda (`adapter.generate` içindeki `:414-421`) kullanıldığını doğrula. Text-only yolda `images`/`images_kwargs` gönderilmez.

- [ ] **Step 1: Failing test** — `backend/tests/test_worker_contract.py`

```python
import sys
from pathlib import Path

WORKER = Path(__file__).resolve().parents[2] / "worker"
sys.path.insert(0, str(WORKER))
from clinfusion_worker import build_chat_messages  # noqa: E402

def test_build_chat_messages_text_only():
    msgs = build_chat_messages([{"role": "user", "text": "önceki"},
                                {"role": "assistant", "text": "cevap"}],
                               prompt="yeni", n_images=0)
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"][-1]["text"] == "yeni"
    assert msgs[-1]["content"][-1]["type"] == "text"

def test_build_chat_messages_with_images():
    msgs = build_chat_messages([], prompt="bak", n_images=2)
    types = [c["type"] for c in msgs[-1]["content"]]
    assert types == ["image", "image", "text"]
```

- [ ] **Step 2: Fail gör**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/test_worker_contract.py -v`
Expected: FAIL

- [ ] **Step 3: `worker/clinfusion_worker.py` yaz**

```python
"""ClinFusion-32B resident HTTP worker (env: clinfusion).

Backend'den bağımsız çalışır; RADAR ile aynı GPU'yu paylaştığı için
yalnız backend kuyruğu üzerinden çağrılmalıdır.
"""
import os
import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

REPO = Path(os.environ.get("CLINFUSION_REPO", "/home/nvidia/ClinFusion"))
MODEL = os.environ.get("CLINFUSION_MODEL", "cache/models/ClinFusion-32B")
PORT = int(os.environ.get("WORKER_PORT", "8100"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "4096"))


def build_chat_messages(history, prompt, n_images):
    msgs = []
    for turn in history:
        msgs.append({"role": turn["role"],
                     "content": [{"type": "text", "text": turn["text"]}]})
    content = [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": prompt})
    msgs.append({"role": "user", "content": content})
    return msgs


class GenerateRequest(BaseModel):
    history: list = []
    prompt: str
    attachment_paths: list = []


class WorkerState:
    def __init__(self):
        self.adapter = None
        self.ready = False
        self.loading = False
        self.error = None

    def load(self):
        self.loading = True
        try:
            os.chdir(REPO)
            sys.path.insert(0, str(REPO))
            from custom_model.medevalkit_adapter_qwen3_vl import MedEvalKitAdapter
            self.adapter = MedEvalKitAdapter(
                model_path=MODEL,
                model_config={"model_type": "custom",
                              "model_definition_path": "custom_model/medevalkit_adapter_qwen3_vl.py"},
                generation_config={"max_new_tokens": MAX_NEW_TOKENS,
                                   "temperature": 0.0, "top_p": 1.0,
                                   "repetition_penalty": 1.0},
            )
            self.ready = True
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
        finally:
            self.loading = False


STATE = WorkerState()
app = FastAPI(title="ClinFusion Worker")


@app.on_event("startup")
def _startup():
    threading.Thread(target=STATE.load, daemon=True).start()


@app.get("/health")
def health():
    return {"ready": STATE.ready, "loading": STATE.loading,
            "error": STATE.error, "model": MODEL}


@app.post("/generate")
def generate(req: GenerateRequest):
    if not STATE.ready:
        return JSONResponse(status_code=503, content={"error": "model yükleniyor"})
    ad = STATE.adapter
    import torch

    volume, processed = ad._pre_sample_slices_from_volume(
        list(req.attachment_paths), ad.processor
    )
    pil_images = [ad._read_pil_image(p) for p in processed]
    pil_images = [i for i in pil_images if i is not None]

    messages = build_chat_messages(req.history, req.prompt, len(pil_images))
    text = ad.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
    if pil_images:
        kwargs["images"] = [pil_images]
        if volume is not None:
            kwargs["images_kwargs"] = {"volume": [volume]}
    inputs = ad.processor(**kwargs).to(ad.device)

    n_in = inputs.input_ids.shape[1]
    gen = {"max_new_tokens": ad.max_new_tokens, "eos_token_id": ad.eos_token_id}
    if ad.temperature > 0:
        gen.update(do_sample=True, temperature=ad.temperature, top_p=ad.top_p)
    with torch.no_grad():
        out = ad.llm.generate(**inputs, **gen)
    reply = ad.processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
    return {"reply": reply}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
```

- [ ] **Step 4: İzole testleri geçir**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests/test_worker_contract.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Worker'ı gerçek modelle başlat (clinfusion env) ve health doğrula**

Run:
```bash
cd /home/nvidia/medportal
nohup /home/nvidia/miniconda3/envs/clinfusion/bin/python worker/clinfusion_worker.py > /tmp/worker.log 2>&1 &
sleep 5
curl -s http://127.0.0.1:8100/health
```
Expected: ilk çağrıda `{"ready": false, "loading": true, ...}`; 32B yüklenince (~10 dk) `"ready": true`. Yükleme bitene kadar `curl` tekrarla.

- [ ] **Step 6: Commit**

```bash
git add worker/clinfusion_worker.py backend/tests/test_worker_contract.py
git commit -m "feat: clinfusion-32b resident http worker"
```

---

## Task 9: Frontend — iskelet + RADAR sekmesi

**Files:**
- Create: `frontend/index.html`, `frontend/styles.css`, `frontend/app.js`

- [ ] **Step 1: `frontend/index.html` yaz**

```html
<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MedPortal</title>
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="/styles.css">
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen">
<header class="border-b border-slate-800 px-6 py-3 flex items-center gap-4">
  <h1 class="font-semibold text-lg">MedPortal</h1>
  <span id="status" class="text-xs text-slate-400"></span>
  <nav class="ml-auto flex gap-2">
    <button id="tab-radar" class="tab-btn tab-active">RADAR</button>
    <button id="tab-chat" class="tab-btn">ClinFusion</button>
  </nav>
</header>

<main class="p-6">
  <section id="view-radar" class="grid grid-cols-1 lg:grid-cols-3 gap-6">
    <div class="lg:col-span-1 space-y-4">
      <div id="drop" class="border-2 border-dashed border-slate-700 rounded-lg p-6 text-center text-sm text-slate-400 cursor-pointer">
        Karın BT (.nii.gz) sürükle-bırak / seç
        <input id="file" type="file" accept=".nii.gz,.nii" class="hidden">
      </div>
      <div class="bg-slate-900 rounded-lg p-2">
        <img id="slice" class="w-full rounded" alt="CT kesit">
        <input id="slice-range" type="range" min="0" max="0" value="0" class="w-full mt-2">
        <div class="text-xs text-slate-500 text-right"><span id="slice-info">-</span></div>
      </div>
    </div>
    <div class="lg:col-span-2 space-y-3">
      <div class="flex items-center gap-3">
        <input id="search" placeholder="Bulgu ara…" class="bg-slate-900 rounded px-3 py-2 text-sm flex-1">
        <label class="text-xs text-slate-400">Eşik <span id="thr-val">0.50</span></label>
        <input id="threshold" type="range" min="0" max="1" step="0.01" value="0.5">
        <button id="csv" class="bg-slate-800 px-3 py-2 rounded text-sm" disabled>CSV indir</button>
      </div>
      <div class="bg-slate-900 rounded-lg overflow-hidden">
        <table class="w-full text-sm">
          <thead class="bg-slate-800 text-slate-300">
            <tr><th class="text-left px-3 py-2">Bulgu</th><th class="text-right px-3 py-2 cursor-pointer" id="sort-score">Skor ▾</th></tr>
          </thead>
          <tbody id="findings"></tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="view-chat" class="hidden max-w-3xl mx-auto">
    <div class="flex justify-end mb-2"><button id="new-chat" class="bg-slate-800 px-3 py-1 rounded text-sm">+ Yeni sohbet</button></div>
    <div id="messages" class="bg-slate-900 rounded-lg p-4 h-[420px] overflow-y-auto space-y-3 text-sm"></div>
    <div id="attachments" class="text-xs text-slate-400 my-2"></div>
    <div class="flex items-center gap-2">
      <label class="bg-slate-800 px-3 py-2 rounded cursor-pointer">📎<input id="chat-file" type="file" accept=".nii.gz,.nii,.jpg,.jpeg,.png" multiple class="hidden"></label>
      <input id="chat-input" placeholder="Mesaj yaz…" class="bg-slate-900 rounded px-3 py-2 flex-1 text-sm">
      <button id="send" class="bg-blue-600 px-4 py-2 rounded text-sm">Gönder</button>
    </div>
  </section>
</main>
<script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: `frontend/styles.css` yaz**

```css
.tab-btn { padding: 6px 14px; border-radius: 6px; background: #1e293b; color: #94a3b8; font-size: 14px; }
.tab-active { background: #2563eb; color: #fff; }
.finding-row { border-top: 1px solid #1e293b; }
.score-high { color: #f87171; } .score-mid { color: #fbbf24; } .score-low { color: #94a3b8; }
```

- [ ] **Step 3: `frontend/app.js` yaz (RADAR + sekmeler; chat Task 10'da genişletilecek)**

```javascript
const $ = (id) => document.getElementById(id);
let radarFindings = [];
let radarUploadId = null;
let sortDesc = true;

async function status() {
  try {
    const s = await (await fetch("/api/status")).json();
    $("status").textContent = `GPU ${s.gpu} · RADAR ${s.radar_ready ? "hazır" : "-"} · ClinFusion ${s.clinfusion_ready ? "hazır" : "yükleniyor"} · ${s.busy ? "meşgul" : "boşta"}`;
  } catch { $("status").textContent = "durum alınamadı"; }
}
setInterval(status, 3000); status();

function showTab(which) {
  $("view-radar").classList.toggle("hidden", which !== "radar");
  $("view-chat").classList.toggle("hidden", which !== "chat");
  $("tab-radar").classList.toggle("tab-active", which === "radar");
  $("tab-chat").classList.toggle("tab-active", which === "chat");
}
$("tab-radar").onclick = () => showTab("radar");
$("tab-chat").onclick = () => showTab("chat");

$("drop").onclick = () => $("file").click();
$("drop").ondragover = (e) => e.preventDefault();
$("drop").ondrop = (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) uploadRadar(e.dataTransfer.files[0]); };
$("file").onchange = () => { if ($("file").files[0]) uploadRadar($("file").files[0]); };

async function uploadRadar(file) {
  const fd = new FormData(); fd.append("file", file);
  const up = await (await fetch("/api/upload", { method: "POST", body: fd })).json();
  if (!up.upload_id) { alert("Yükleme hatası: " + JSON.stringify(up)); return; }
  radarUploadId = up.upload_id;
  $("slice-range").max = up.n_slices - 1;
  $("slice-range").value = Math.floor(up.n_slices / 2);
  loadSlice();
  const j = await (await fetch("/api/radar", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ upload_id: up.upload_id }) })).json();
  pollRadar(j.job_id);
}

async function loadSlice() {
  if (!radarUploadId) return;
  const idx = $("slice-range").value;
  const r = await fetch(`/api/nifti/${radarUploadId}/slice?idx=${idx}`);
  $("slice").src = URL.createObjectURL(await r.blob());
  $("slice-info").textContent = `${idx} / ${r.headers.get("X-Total-Slices")}`;
}
$("slice-range").oninput = loadSlice;

async function pollRadar(jobId) {
  for (;;) {
    const j = await (await fetch(`/api/radar/${jobId}`)).json();
    if (j.status === "error") { alert("RADAR hatası: " + j.error); return; }
    if (j.status === "done") {
      radarFindings = j.findings;
      renderFindings();
      $("csv").disabled = false;
      $("csv").onclick = () => window.open(j.csv_url);
      return;
    }
    await new Promise(r => setTimeout(r, 1500));
  }
}

function renderFindings() {
  const thr = parseFloat($("threshold").value);
  const q = $("search").value.toLowerCase();
  let rows = radarFindings.filter(f => f.score >= thr && f.name.toLowerCase().includes(q));
  rows.sort((a, b) => sortDesc ? b.score - a.score : a.score - b.score);
  $("findings").innerHTML = rows.map(f =>
    `<tr class="finding-row"><td class="px-3 py-2">${f.name}</td>
     <td class="text-right px-3 py-2 ${f.score >= 0.7 ? "score-high" : f.score >= 0.3 ? "score-mid" : "score-low"}">${f.score.toFixed(3)}</td></tr>`
  ).join("") || `<tr><td class="px-3 py-3 text-slate-500" colspan="2">Eşiği geçen bulgu yok</td></tr>`;
}
$("threshold").oninput = () => { $("thr-val").textContent = parseFloat($("threshold").value).toFixed(2); renderFindings(); };
$("search").oninput = renderFindings;
$("sort-score").onclick = () => { sortDesc = !sortDesc; $("sort-score").textContent = sortDesc ? "Skor ▾" : "Skor ▴"; renderFindings(); };
```

- [ ] **Step 4: Backend'i geçici başlat ve RADAR sekmesini tarayıcıda doğrula**

Run:
```bash
cd /home/nvidia/medportal
nohup /home/nvidia/miniconda3/envs/medportal/bin/python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8080 > /tmp/backend.log 2>&1 &
sleep 3
curl -s http://127.0.0.1:8080/api/status
```
Expected: JSON durum. (Worker kapalıysa `clinfusion_ready:false` normal.) Tarayıcıda `http://192.168.1.162:8080` açıp demo `.nii.gz` (ör. `~/radar/data/demo_cases/AC423ccbe.nii.gz`) yükle; kesit görüntüsü + bulgu tablosu görünmeli.

- [ ] **Step 5: Commit**

```bash
git add frontend/index.html frontend/styles.css frontend/app.js
git commit -m "feat: frontend skeleton and radar tab"
```

---

## Task 10: Frontend — ClinFusion sohbet sekmesi

**Files:**
- Modify: `frontend/app.js` (sonuna chat mantığı ekle)

- [ ] **Step 1: `frontend/app.js` sonuna ekle**

```javascript
let chatHistory = [];
let chatAttachments = [];

$("chat-file").onchange = async () => {
  chatAttachments = [];
  for (const f of $("chat-file").files) {
    const fd = new FormData(); fd.append("file", f);
    const up = await (await fetch("/api/upload", { method: "POST", body: fd })).json();
    if (up.upload_id) chatAttachments.push({ id: up.upload_id, name: up.filename });
  }
  $("attachments").textContent = chatAttachments.map(a => "📎 " + a.name).join("  ");
};

function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = role === "user"
    ? "bg-slate-800 rounded-lg p-3 max-w-[80%]"
    : "bg-blue-900/60 rounded-lg p-3 ml-auto max-w-[85%] whitespace-pre-wrap";
  div.textContent = text;
  $("messages").appendChild(div);
  $("messages").scrollTop = $("messages").scrollHeight;
}

$("new-chat").onclick = async () => {
  await fetch("/api/chat/reset", { method: "POST" });
  chatHistory = []; chatAttachments = [];
  $("messages").innerHTML = ""; $("attachments").textContent = "";
};

async function send() {
  const prompt = $("chat-input").value.trim();
  if (!prompt) return;
  addMsg("user", prompt);
  $("chat-input").value = "";
  const body = { prompt, history: chatHistory, attachment_ids: chatAttachments.map(a => a.id) };
  chatAttachments = []; $("attachments").textContent = "";
  const j = await (await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
  const thinking = document.createElement("div");
  thinking.textContent = "… düşünüyor (32B)";
  thinking.className = "text-slate-500 text-xs";
  $("messages").appendChild(thinking);
  for (;;) {
    const r = await (await fetch(`/api/chat/${j.job_id}`)).json();
    if (r.status === "error") { thinking.textContent = "Hata: " + r.error; return; }
    if (r.status === "done") {
      thinking.remove();
      addMsg("assistant", r.reply);
      chatHistory.push({ role: "user", text: prompt });
      chatHistory.push({ role: "assistant", text: r.reply });
      return;
    }
    await new Promise(r => setTimeout(r, 2000));
  }
}
$("send").onclick = send;
$("chat-input").addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
```

- [ ] **Step 2: Tarayıcıda sohbeti doğrula**

Worker çalışır durumda (`/health` → ready:true) iken `http://192.168.1.162:8080` → ClinFusion sekmesi → metin sorusu gönder; cevap görünmeli. Sonra `.nii.gz` ekleyip 3D soru sor; cevap gelmeli. "Yeni sohbet" sıfırlamalı.

- [ ] **Step 3: Commit**

```bash
git add frontend/app.js
git commit -m "feat: clinfusion chat tab with multi-turn"
```

---

## Task 11: `run.sh`, duman testi, doküman

**Files:**
- Create: `run.sh`, `scripts/smoke.sh`, `KULLANIM.md`
- Modify: `README.md` (çalıştırma bölümü ekle)

- [ ] **Step 1: `run.sh` yaz**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

WORKER_PORT="${WORKER_PORT:-8100}"
BACKEND_PORT="${BACKEND_PORT:-8080}"

pkill -f clinfusion_worker.py 2>/dev/null || true
pkill -f "uvicorn.*backend.app.main" 2>/dev/null || true

# Worker supervisor: düşerse yeniden başlatır (SSH oturumu kapansa da yaşar)
setsid nohup bash -c '
  while true; do
    echo "[worker] starting $(date)"
    /home/nvidia/miniconda3/envs/clinfusion/bin/python worker/clinfusion_worker.py >> logs/worker.log 2>&1 || true
    echo "[worker] exited; 10s sonra yeniden"
    sleep 10
  done
' >/dev/null 2>&1 &
echo "worker supervisor başlatıldı"

echo "32B yükleniyor (health bekleniyor)…"
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${WORKER_PORT}/health" | grep -q '"ready": *true'; then
    echo "worker hazır."; break
  fi
  sleep 10
done

echo "[backend] başlatılıyor…"
nohup /home/nvidia/miniconda3/envs/medportal/bin/python -m uvicorn \
  backend.app.main:app --host 0.0.0.0 --port "$BACKEND_PORT" >> logs/backend.log 2>&1 &
echo "backend PID: $!"

echo "Erişim: http://192.168.1.162:${BACKEND_PORT}"
echo "Model durumu: curl -s http://127.0.0.1:${WORKER_PORT}/health"
```

Run: `mkdir -p logs && chmod +x run.sh`

- [ ] **Step 2: `scripts/smoke.sh` yaz**

```bash
#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8080}"
NII="${NII:-/home/nvidia/radar/data/demo_cases/AC423ccbe.nii.gz}"

echo "== status =="; curl -sf "$BASE/api/status"; echo
echo "== upload =="
UP=$(curl -sf -F "file=@${NII}" "$BASE/api/upload")
echo "$UP"
UID=$(echo "$UP" | python3 -c "import sys,json;print(json.load(sys.stdin)['upload_id'])")
echo "== slice =="; curl -sf -o /tmp/smoke_slice.png -D /tmp/smoke_h.txt "$BASE/api/nifti/${UID}/slice?idx=10"; grep -i x-total-slices /tmp/smoke_h.txt
echo "== radar =="
J=$(curl -sf -H 'Content-Type: application/json' -d "{\"upload_id\":\"${UID}\"}" "$BASE/api/radar")
JID=$(echo "$J" | python3 -c "import sys,json;print(json.load(sys.stdin)['job_id'])")
for i in $(seq 1 120); do
  R=$(curl -sf "$BASE/api/radar/${JID}")
  ST=$(echo "$R" | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])")
  [ "$ST" = "done" ] && { echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);print('findings:',len(d['findings']),'top:',d['findings'][0]['name'])"; break; }
  [ "$ST" = "error" ] && { echo "RADAR error: $R"; exit 1; }
  sleep 2
done
echo "== backend testleri =="
cd "$(dirname "$0")/.."
/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests -q
echo "SMOKE OK"
```

Run: `chmod +x scripts/smoke.sh`

- [ ] **Step 3: Uçtan uca duman testi çalıştır**

Run:
```bash
cd /home/nvidia/medportal
./scripts/smoke.sh
```
Expected: status JSON, upload `upload_id`, `X-Total-Slices`, `findings: 146 top: <bulgu>`, ardından testler `passed`, `SMOKE OK`.

- [ ] **Step 4: `KULLANIM.md` yaz** (kısa kullanım + mimari + sorun giderme; RADAR/ClinFusion README'lerine atıf)

- [ ] **Step 5: Commit**

```bash
git add run.sh scripts/smoke.sh KULLANIM.md README.md
git commit -m "feat: run script, smoke test, usage docs"
```

---

## Task 12: Son doğrulama

- [ ] **Step 1: Tüm backend testleri**

Run: `/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests -v`
Expected: hepsi PASS.

- [ ] **Step 2: `run.sh` ile sıfırdan başlat, tarayıcıda iki sekmeyi doğrula**

RADAR: kesit + tablo + eşik + CSV. ClinFusion: metin + 3D eki + çok turlu + yeni sohbet.

- [ ] **Step 3: Spec kabul kriterleri kontrol listesi**

- [ ] RADAR 146 bulgu + eşik/sıralama/CSV
- [ ] NIfTI aksiyel görüntüleyici
- [ ] ClinFusion 32B resident, çok turlu + yeni sohbet
- [ ] Tek GPU kuyruğu (aynı anda tek iş)
- [ ] `.nii.gz` doğrulama + boyut sınırı
- [ ] `http://192.168.1.162:8080` erişilebilir

- [ ] **Step 4: Final commit**

```bash
git add -A && git commit -m "chore: final verification" || true
```

---

## Notlar / Riskler
- **32B ilk yükleme ~10 dk**: `run.sh` worker'ı önce başlatır; backend `/api/status` `clinfusion_ready:false` gösterirken sohbet kuyruğa alınır ama worker hazır olana dek hata dönebilir. İstersen sohbet endpoint'i worker `ready` olana kadar bekleyebilir (opsiyonel iyileştirme).
- **Bellek**: 32B resident ~89GB; RADAR yüklemesi ~2GB. Aynı anda çalışmaz (kuyruk serileştirir).
- **Worker ömrü**: `run.sh` öldürüp yeniden başlatır; geliştirme sırasında `MAX_NEW_TOKENS` düşürülebilir.
- **Test edilebilirlik**: Backend testleri GPU gerektirmez (subprocess/worker mock'lu).
