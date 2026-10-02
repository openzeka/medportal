import asyncio
import csv

import pytest

from app import config
from app.radar import build_command, parse_csv, run_radar


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


def test_parse_csv_header_only(tmp_path):
    p = tmp_path / "empty.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerow(["file_name", "肝_硬化 (Liver_Cirrhosis)"])
    with pytest.raises(RuntimeError, match="RADAR CSV is empty"):
        parse_csv(p)


def test_parse_csv_missing_columns(tmp_path):
    p = tmp_path / "short.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "肝_硬化 (Liver_Cirrhosis)", "肺_结节 (Lung_Nodule)"])
        w.writerow(["x.nii.gz", "0.9"])
    with pytest.raises(RuntimeError, match="column mismatch"):
        parse_csv(p)


def test_parse_csv_extra_columns(tmp_path):
    p = tmp_path / "long.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "肝_硬化 (Liver_Cirrhosis)"])
        w.writerow(["x.nii.gz", "0.9", "0.1", "0.2", "0.3"])
    with pytest.raises(RuntimeError, match="column mismatch"):
        parse_csv(p)


def test_parse_csv_invalid_score(tmp_path):
    p = tmp_path / "bad.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "肝_硬化 (Liver_Cirrhosis)"])
        w.writerow(["x.nii.gz", "abc"])
    with pytest.raises(RuntimeError, match="invalid score"):
        parse_csv(p)


def test_build_command(tmp_path):
    cmd = build_command("radar", tmp_path / "in", tmp_path / "out")
    assert cmd[0] == config.RADAR_PY
    assert cmd[1] == config.RADAR_SCRIPT
    assert "--img_dir" in cmd and "--save_tag" in cmd


def _fake_script(tmp_path, body):
    script = tmp_path / "fake_radar.py"
    script.write_text(body, encoding="utf-8")
    return script


def _make_input(tmp_path):
    nifti = tmp_path / "case.nii.gz"
    nifti.write_bytes(b"fake-nifti")
    return nifti


def _configure(monkeypatch, script):
    monkeypatch.setattr(config, "RADAR_PY", __import__("sys").executable)
    monkeypatch.setattr(config, "RADAR_SCRIPT", str(script))


_WRITE_CSV = """
import argparse, os, csv, sys
p = argparse.ArgumentParser()
p.add_argument("--img_dir")
p.add_argument("--save_dir")
p.add_argument("--save_tag")
a = p.parse_args()
os.makedirs(a.save_dir, exist_ok=True)
with open(os.path.join(a.save_dir, "RADAR_infer_results_%s.csv" % a.save_tag), "w",
          encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file_name", "\u809d_\u786c\u5316 (Liver_Cirrhosis)", "\u80ba_\u7ed3\u8282 (Lung_Nodule)"])
    w.writerow(["case.nii.gz", "0.1", "0.9"])
"""


async def test_run_radar_success(tmp_path, monkeypatch):
    script = _fake_script(tmp_path, _WRITE_CSV)
    _configure(monkeypatch, script)
    job_dir = tmp_path / "job"
    result = await run_radar(_make_input(tmp_path), job_dir, "jid1")
    assert result["csv_name"] == "RADAR_infer_results_jid1.csv"
    names = [f["name"] for f in result["findings"]]
    assert names == ["Lung_Nodule", "Liver_Cirrhosis"]
    assert (job_dir / "in" / "case.nii.gz").is_file()
    assert (job_dir / "job.log").is_file()


async def test_run_radar_nonzero_exit(tmp_path, monkeypatch):
    script = _fake_script(
        tmp_path,
        "import sys; sys.stderr.write('BOOM\\n'); sys.exit(1)\n",
    )
    _configure(monkeypatch, script)
    with pytest.raises(RuntimeError, match="RADAR failed") as exc:
        await run_radar(_make_input(tmp_path), tmp_path / "job", "jid2")
    assert "BOOM" in str(exc.value)


async def test_run_radar_missing_csv(tmp_path, monkeypatch):
    script = _fake_script(tmp_path, "import sys; sys.exit(0)\n")
    _configure(monkeypatch, script)
    with pytest.raises(RuntimeError, match="produced no CSV"):
        await run_radar(_make_input(tmp_path), tmp_path / "job", "jid3")


async def test_run_radar_header_only_csv(tmp_path, monkeypatch):
    script = _fake_script(
        tmp_path,
        _WRITE_CSV.replace(
            '    w.writerow(["case.nii.gz", "0.1", "0.9"])\n', ""
        ),
    )
    _configure(monkeypatch, script)
    with pytest.raises(RuntimeError, match="RADAR CSV is empty"):
        await run_radar(_make_input(tmp_path), tmp_path / "job", "jid4")


async def test_run_radar_timeout_kills_process_group(tmp_path, monkeypatch):
    marker = tmp_path / "child_alive.txt"
    body = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c',\n"
        "    'import time,sys; time.sleep(2); open(sys.argv[1], \"w\").write(\"x\")',\n"
        f"    r'{marker}'])\n"
        "time.sleep(30)\n"
    )
    script = _fake_script(tmp_path, body)
    _configure(monkeypatch, script)
    monkeypatch.setattr(config, "RADAR_TIMEOUT", 1)
    with pytest.raises(RuntimeError, match="RADAR timed out"):
        await run_radar(_make_input(tmp_path), tmp_path / "job", "jid5")
    await asyncio.sleep(2.5)
    assert not marker.exists()
