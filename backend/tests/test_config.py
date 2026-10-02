import importlib

from app import config


def test_paths_are_absolute():
    assert config.ROOT.is_absolute()
    assert config.UPLOADS.is_absolute()
    assert config.JOBS.is_absolute()
    assert config.FRONTEND.is_absolute()


def test_defaults(monkeypatch):
    for key in ("WORKER_HOST", "WORKER_PORT"):
        monkeypatch.delenv(key, raising=False)
    reloaded = importlib.reload(config)
    try:
        assert reloaded.MAX_UPLOAD_MB == 500
        assert reloaded.RADAR_TIMEOUT == 600
        assert reloaded.CHAT_TIMEOUT == 1800
        assert reloaded.WORKER_URL.startswith("http://127.0.0.1")
    finally:
        importlib.reload(config)


def test_ensure_dirs_creates_workspace_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(config, "JOBS", tmp_path / "jobs")
    config.ensure_dirs()
    assert (tmp_path / "uploads").is_dir()
    assert (tmp_path / "jobs").is_dir()
