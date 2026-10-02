import io
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

# Default window: WC 0 / WW 2000 (identical to the legacy WINDOW_MIN=-1000 / WINDOW_MAX=1000)
DEFAULT_WC = 0.0
DEFAULT_WW = 2000.0
WINDOW_MIN = -1000.0  # kept for backward compatibility
WINDOW_MAX = 1000.0


def load_volume(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data


@lru_cache(maxsize=4)
def _load_volume_cached(key: tuple) -> np.ndarray:
    return load_volume(Path(key[0]))


def load_volume_cached(path: Path) -> np.ndarray:
    p = Path(path)
    try:
        mtime = p.stat().st_mtime_ns
    except OSError:
        mtime = None
    return _load_volume_cached((str(p), mtime))


def n_slices(path: Path) -> int:
    return int(load_volume_cached(path).shape[2])


def _to_uint8(sl: np.ndarray, wc: float = DEFAULT_WC, ww: float = DEFAULT_WW) -> np.ndarray:
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    sl = np.clip(sl, lo, hi)
    sl = (sl - lo) / (hi - lo)
    return np.rint(sl * 255.0).astype(np.uint8)


def slice_png(path: Path, idx: int, wc: float = None, ww: float = None) -> bytes:
    """Render an axial slice as PNG with a WC/WW window.

    wc=None / ww=None -> default WC 0 / WW 2000 (legacy behavior).
    ww <= 0 → ValueError.
    """
    wc = DEFAULT_WC if wc is None else float(wc)
    ww = DEFAULT_WW if ww is None else float(ww)
    if ww <= 0:
        raise ValueError("ww must be positive")
    vol = load_volume_cached(path)
    idx = max(0, min(idx, vol.shape[2] - 1))
    # axial slice (z axis); rotate 90° to keep the image upright
    sl = np.rot90(vol[:, :, idx])
    arr = _to_uint8(sl, wc, ww)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()