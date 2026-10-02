import re
import uuid
from pathlib import Path

from . import config


class UploadError(Exception):
    pass


UPLOAD_ID_RE = re.compile(r"[0-9a-f]{12}")


def _kind(name: str) -> str:
    lower = name.lower()
    if lower.endswith(config.ALLOWED_NIFTI):
        return "nifti"
    if lower.endswith(config.ALLOWED_IMAGE):
        return "image"
    raise UploadError(f"unsupported file type: {name}")


def validate_filename(filename: str) -> str:
    return _kind(filename)


def save_upload(filename: str, data: bytes) -> dict:
    kind = _kind(filename)
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise UploadError("file too large")
    uid = uuid.uuid4().hex[:12]
    dest_dir = config.UPLOADS / uid
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = Path(filename).name
    (dest_dir / safe).write_bytes(data)
    return {"upload_id": uid, "kind": kind, "filename": safe, "path": dest_dir / safe}


def resolve(upload_id: str) -> Path:
    if not UPLOAD_ID_RE.fullmatch(upload_id):
        raise UploadError("invalid upload_id")
    d = config.UPLOADS / upload_id
    if not d.is_dir():
        raise UploadError("upload_id not found")
    files = [p for p in d.iterdir() if p.is_file()]
    if not files:
        raise UploadError("upload is empty")
    if len(files) > 1:
        raise UploadError("ambiguous upload")
    return files[0]
