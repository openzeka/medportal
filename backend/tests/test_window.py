"""WC/WW window support tests (unit + API)."""
import io
import os
import tempfile

import nibabel as nib
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app import config, store as store_mod
from app.nifti import slice_png
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


def _make_nii(tmp_path, value=500.0, name="v.nii.gz"):
    data = np.full((4, 5, 6), value, dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    p = tmp_path / name
    nib.save(img, str(p))
    return p


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


def _pixel(png):
    return Image.open(io.BytesIO(png)).getpixel((0, 0))


def test_default_window_matches_legacy(tmp_path):
    p = _make_nii(tmp_path, -321.0)
    assert slice_png(p, 2) == slice_png(p, 2, wc=0, ww=2000)


def test_window_changes_pixel_mapping(tmp_path):
    p = _make_nii(tmp_path, 500.0)
    # Full window (0/2000): (500+1000)/2000*255 ≈ 191
    assert _pixel(slice_png(p, 0)) == 191
    # Bone window (400/1800): (500-(-500))/1800*255 ≈ 142
    assert _pixel(slice_png(p, 0, wc=400, ww=1800)) == 142


def test_window_clips_out_of_range(tmp_path):
    # Separate filenames: load_volume_cached keys on (path, mtime), so writing a
    # second volume to the same path within one mtime tick would return the stale one.
    p = _make_nii(tmp_path, 3000.0, "high.nii.gz")
    assert _pixel(slice_png(p, 0)) == 255            # above window -> white
    p2 = _make_nii(tmp_path, -3000.0, "low.nii.gz")
    assert _pixel(slice_png(p2, 0)) == 0             # below window -> black


def test_bad_window_rejected(tmp_path):
    p = _make_nii(tmp_path)
    with pytest.raises(ValueError):
        slice_png(p, 0, ww=0)
    with pytest.raises(ValueError):
        slice_png(p, 0, ww=-100)


def test_api_slice_window_params():
    r = client.post("/api/upload", files={"file": ("v.nii.gz", _nii_bytes())})
    uid = r.json()["upload_id"]
    a = client.get(f"/api/nifti/{uid}/slice?idx=1&wc=40&ww=400")
    assert a.status_code == 200
    assert a.headers["X-Total-Slices"] == "6"
    b = client.get(f"/api/nifti/{uid}/slice?idx=1")
    assert a.content != b.content


def test_api_slice_bad_window_400():
    r = client.post("/api/upload", files={"file": ("v.nii.gz", _nii_bytes())})
    uid = r.json()["upload_id"]
    assert client.get(f"/api/nifti/{uid}/slice?idx=1&ww=0").status_code == 400
    assert client.get(f"/api/nifti/{uid}/slice?idx=1&wc=abc&ww=400").status_code == 422