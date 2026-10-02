"""RADAR transient-OOM retry behavior tests."""
import csv

import pytest

from app import radar


CSV_HEADER = ["file_name", "心 (Heart_Cardiomegaly)", "肺 (Lung_Nodule)"]
CSV_ROW = ["v.nii.gz", "0.9", "0.1"]


def _write_csv(out_dir, tag):
    p = out_dir / f"RADAR_infer_results_{tag}.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerow(CSV_ROW)


def _paths_from_cmd(cmd):
    out_dir = cmd[cmd.index("--save_dir") + 1]
    tag = cmd[cmd.index("--save_tag") + 1]
    return out_dir, tag


async def test_retries_once_on_oom_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(radar, "OOM_RETRY_DELAY", 0)
    src = tmp_path / "v.nii.gz"
    src.write_bytes(b"x" * 10)
    calls = []

    async def fake_once(cmd, log_path, mode="w"):
        calls.append(mode)
        with open(log_path, mode) as f:
            if len(calls) == 1:
                f.write("torch.AcceleratorError: CUDA error: out of memory\n")
        if len(calls) > 1:
            out_dir, tag = _paths_from_cmd(cmd)
            _write_csv(__import__("pathlib").Path(out_dir), tag)
        return 1 if len(calls) == 1 else 0

    monkeypatch.setattr(radar, "_run_once", fake_once)
    res = await radar.run_radar(src, tmp_path / "job", "abc")
    assert len(calls) == 2
    assert calls == ["w", "a"]
    assert res["findings"][0]["name"] == "Heart_Cardiomegaly"


async def test_no_retry_on_non_oom_error(tmp_path, monkeypatch):
    monkeypatch.setattr(radar, "OOM_RETRY_DELAY", 0)
    src = tmp_path / "v.nii.gz"
    src.write_bytes(b"x" * 10)
    calls = []

    async def fake_once(cmd, log_path, mode="w"):
        calls.append(mode)
        with open(log_path, mode) as f:
            f.write("KeyError: 'checkpoint'\n")
        return 1

    monkeypatch.setattr(radar, "_run_once", fake_once)
    with pytest.raises(RuntimeError, match="exit 1"):
        await radar.run_radar(src, tmp_path / "job", "abc")
    assert len(calls) == 1


async def test_gives_up_after_max_attempts_on_persistent_oom(tmp_path, monkeypatch):
    monkeypatch.setattr(radar, "OOM_RETRY_DELAY", 0)
    src = tmp_path / "v.nii.gz"
    src.write_bytes(b"x" * 10)
    calls = []

    async def fake_once(cmd, log_path, mode="w"):
        calls.append(mode)
        with open(log_path, mode) as f:
            f.write("CUDA error: out of memory\n")
        return 1

    monkeypatch.setattr(radar, "_run_once", fake_once)
    with pytest.raises(RuntimeError, match="out of memory"):
        await radar.run_radar(src, tmp_path / "job", "abc")
    assert len(calls) == 2