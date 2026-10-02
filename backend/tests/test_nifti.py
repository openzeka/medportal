import io
import os

import numpy as np
import nibabel as nib
from PIL import Image

from app.nifti import load_volume, load_volume_cached, n_slices, slice_png


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
    assert n_slices(p) == 6


def test_slice_png(tmp_path):
    p = _make_nii(tmp_path)
    png = slice_png(p, 3)
    assert png[:4] == b"\x89PNG"
    img = Image.open(io.BytesIO(png))
    assert img.size == (4, 5)


def test_slice_png_clamps_index(tmp_path):
    p = _make_nii(tmp_path)
    assert slice_png(p, 999) == slice_png(p, 5)


def test_load_volume_cached_reads_disk_once(tmp_path, monkeypatch):
    p = _make_nii(tmp_path)
    calls = []
    orig = nib.Nifti1Image.get_fdata

    def counting(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(nib.Nifti1Image, "get_fdata", counting)
    load_volume_cached(p)
    load_volume_cached(p)
    assert len(calls) == 1


def test_load_volume_cached_refreshes_on_mtime_change(tmp_path, monkeypatch):
    p = _make_nii(tmp_path)
    calls = []
    orig = nib.Nifti1Image.get_fdata

    def counting(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(nib.Nifti1Image, "get_fdata", counting)
    load_volume_cached(p)
    st = p.stat()
    os.utime(str(p), ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    load_volume_cached(p)
    assert len(calls) == 2
