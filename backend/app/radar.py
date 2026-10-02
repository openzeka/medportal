import asyncio
import csv
import os
import re
import shutil
import signal
from pathlib import Path

from . import config

# Memory peaks right after the 32B worker finishes loading can make the RADAR
# subprocess hit a transient CUDA OOM; retry once after a short wait.
OOM_RETRY_DELAY = 30.0
OOM_MAX_ATTEMPTS = 2
_OOM_RE = re.compile(r"out of memory", re.IGNORECASE)


def build_command(tag: str, in_dir: Path, out_dir: Path):
    return [
        config.RADAR_PY,
        config.RADAR_SCRIPT,
        "--img_dir", str(in_dir),
        "--save_dir", str(out_dir),
        "--save_tag", tag,
    ]


def _kill_group(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


_EN = re.compile(r"\(([^)]+)\)\s*$")


def parse_csv(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError("RADAR CSV is empty")
        try:
            row = next(reader)
        except StopIteration:
            raise RuntimeError("RADAR CSV is empty")

    if len(row) != len(header):
        raise RuntimeError(f"RADAR CSV column mismatch: {len(row)}/{len(header)}")

    findings = []
    for label, value in zip(header[1:], row[1:]):
        m = _EN.search(label)
        name = m.group(1) if m else label
        try:
            score = float(value)
        except (TypeError, ValueError):
            raise RuntimeError(f"invalid score {value!r} ({label})")
        findings.append({"name": name, "label": label, "score": score})
    findings.sort(key=lambda x: x["score"], reverse=True)
    return findings


async def _run_once(cmd, log_path: Path, mode: str = "w") -> int:
    """Run the RADAR subprocess; return its exit code (output goes to the log file)."""
    with open(log_path, mode) as logf:
        if mode == "a":
            logf.write("\n--- retrying (CUDA OOM) ---\n")
            logf.flush()
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=logf,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(Path(config.RADAR_SCRIPT).parent),
                start_new_session=True,
                env=env,
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"failed to start RADAR: {e}")

        try:
            return await asyncio.wait_for(proc.wait(), config.RADAR_TIMEOUT)
        except asyncio.TimeoutError:
            _kill_group(proc)
            await proc.wait()
            raise RuntimeError("RADAR timed out")
        except BaseException:
            _kill_group(proc)
            await proc.wait()
            raise


async def run_radar(nifti_path: Path, job_dir: Path, job_id: str):
    in_dir = job_dir / "in"
    out_dir = job_dir / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(nifti_path, in_dir / nifti_path.name)
    log_path = job_dir / "job.log"

    cmd = build_command(job_id, in_dir, out_dir)
    last_tail = ""
    for attempt in range(OOM_MAX_ATTEMPTS):
        code = await _run_once(cmd, log_path, mode="w" if attempt == 0 else "a")
        if code == 0:
            break
        last_tail = log_path.read_text(errors="replace")[-2000:]
        if attempt + 1 < OOM_MAX_ATTEMPTS and _OOM_RE.search(last_tail):
            await asyncio.sleep(OOM_RETRY_DELAY)
            continue
        raise RuntimeError(f"RADAR failed (exit {code}):\n{last_tail}")

    csv_path = out_dir / f"RADAR_infer_results_{job_id}.csv"
    if not csv_path.is_file():
        raise RuntimeError("RADAR produced no CSV")
    findings = parse_csv(csv_path)
    return {"findings": findings, "csv_name": csv_path.name}