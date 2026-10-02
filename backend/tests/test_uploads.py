import pytest
from app import config
from app.uploads import save_upload, resolve, validate_filename, UploadError


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


def test_validate_filename_returns_kind():
    assert validate_filename("scan.nii.gz") == "nifti"
    assert validate_filename("x.PNG") == "image"


def test_validate_filename_rejects_bad_ext():
    with pytest.raises(UploadError):
        validate_filename("evil.exe")


def test_reject_too_big(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)
    with pytest.raises(UploadError):
        save_upload("scan.nii.gz", b"x" * 10)


def test_image_kind_uppercase():
    meta = save_upload("x.PNG", b"..")
    assert meta["kind"] == "image"


def test_resolve_happy_path():
    meta = save_upload("scan.nii.gz", b"x" * 100)
    assert resolve(meta["upload_id"]) == meta["path"]


def test_resolve_unknown_id():
    with pytest.raises(UploadError, match="not found"):
        resolve("0123456789ab")


def test_resolve_rejects_traversal():
    decoy = config.UPLOADS.parent / "secret"
    decoy.mkdir()
    (decoy / "passwd").write_bytes(b"x")
    with pytest.raises(UploadError, match="invalid"):
        resolve("../secret")


def test_resolve_rejects_non_hex_id():
    with pytest.raises(UploadError, match="invalid"):
        resolve("z" * 12)


def test_resolve_empty_dir():
    (config.UPLOADS / ("0" * 12)).mkdir()
    with pytest.raises(UploadError, match="empty"):
        resolve("0" * 12)


def test_resolve_rejects_multiple_files():
    meta = save_upload("scan.nii.gz", b"x" * 100)
    (config.UPLOADS / meta["upload_id"] / "extra.nii").write_bytes(b"y")
    with pytest.raises(UploadError, match="ambiguous"):
        resolve(meta["upload_id"])
