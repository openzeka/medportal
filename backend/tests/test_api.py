import os
import tempfile
import time
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
    monkeypatch.setattr(config, "FRONTEND", tmp_path / "frontend")
    config.UPLOADS.mkdir(parents=True, exist_ok=True)
    config.JOBS.mkdir(parents=True, exist_ok=True)
    store_mod.STORE._jobs.clear()
    yield

client = TestClient(app)

def _nii_bytes():
    data = np.zeros((4, 5, 6), dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    fd, path = tempfile.mkstemp(suffix=".nii.gz")
    os.close(fd)
    try:
        nib.save(img, path)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)

def wait_done(path, timeout=5):
    deadline = time.time() + timeout
    data = client.get(path).json()
    while time.time() < deadline:
        data = client.get(path).json()
        if data["status"] in ("done", "error"):
            return data
        time.sleep(0.02)
    return data

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

def test_upload_bad_ext():
    r = client.post("/api/upload", files={"file": ("v.txt", b"hello")})
    assert r.status_code == 400


def test_upload_dot_filename_rejected():
    r = client.post("/api/upload", files={"file": (".", b"hello")})
    assert r.status_code == 400


def test_upload_size_limit_rejected_before_read(monkeypatch):
    from starlette.datastructures import UploadFile

    reads = []

    async def fail_read(self):
        reads.append(1)
        raise AssertionError("read must not be called")

    monkeypatch.setattr(UploadFile, "read", fail_read)
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)
    r = client.post("/api/upload", files={"file": ("v.nii.gz", b"hello")})
    assert r.status_code == 400
    assert reads == []


def test_radar_rejects_non_nifti():
    up = client.post("/api/upload", files={"file": ("v.png", b"\x89PNG\r\n\x1a\n")}).json()
    assert up["kind"] == "image"
    r = client.post("/api/radar", json={"upload_id": up["upload_id"]})
    assert r.status_code == 400

def test_radar_job(monkeypatch):
    async def fake_run(nifti_path, job_dir, job_id):
        return {"findings": [{"name": "X", "label": "X", "score": 0.9}], "csv_name": "c.csv"}
    import app.main as main
    monkeypatch.setattr(main, "run_radar", fake_run)
    up = client.post("/api/upload", files={"file": ("v.nii.gz", _nii_bytes())}).json()
    jid = client.post("/api/radar", json={"upload_id": up["upload_id"]}).json()["job_id"]
    data = wait_done(f"/api/radar/{jid}")
    assert data["status"] == "done"
    assert data["findings"] == [{"name": "X", "label": "X", "score": 0.9}]
    assert data["csv_url"] == f"/api/radar/{jid}/csv"

def test_radar_error(monkeypatch):
    async def fake_run(nifti_path, job_dir, job_id):
        raise RuntimeError("boom")
    import app.main as main
    monkeypatch.setattr(main, "run_radar", fake_run)
    up = client.post("/api/upload", files={"file": ("v.nii.gz", _nii_bytes())}).json()
    jid = client.post("/api/radar", json={"upload_id": up["upload_id"]}).json()["job_id"]
    data = wait_done(f"/api/radar/{jid}")
    assert data["status"] == "error"
    assert "boom" in data["error"]

def test_radar_csv_404():
    r = client.get("/api/radar/nope/csv")
    assert r.status_code == 404

def test_chat_job(monkeypatch):
    import app.main as main
    async def fake_generate(history, prompt, attachment_paths):
        return "answer"
    monkeypatch.setattr(main.cf, "generate", fake_generate)
    jid = client.post("/api/chat", json={"prompt": "selam"}).json()["job_id"]
    data = wait_done(f"/api/chat/{jid}")
    assert data["status"] == "done"
    assert data["reply"] == "answer"

def test_unknown_job_404():
    assert client.get("/api/radar/nope").status_code == 404
    assert client.get("/api/chat/nope").status_code == 404
