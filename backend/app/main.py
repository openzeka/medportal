import asyncio
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import clinfusion as cf
from . import config, nifti
from .gpu_queue import QUEUE
from .radar import run_radar
from .schemas import ChatRequest, JobIdResponse, RadarRequest, StatusResponse, UploadResponse
from .store import STORE
from .uploads import UploadError, resolve, save_upload, validate_filename

app = FastAPI(title="MedPortal")


@app.on_event("startup")
async def _startup():
    config.ensure_dirs()


_TASKS: set = set()


def _spawn(coro):
    t = asyncio.create_task(coro)
    _TASKS.add(t)
    t.add_done_callback(_TASKS.discard)


async def _radar_task(job_id: str, nifti_path: Path):
    try:
        async with QUEUE.lock(job_id):
            STORE.set_running(job_id)
            job_dir = config.JOBS / job_id
            result = await run_radar(nifti_path, job_dir, job_id)
            STORE.set_done(job_id, result)
    except Exception as e:
        STORE.set_error(job_id, str(e))


async def _chat_task(job_id: str, history, prompt, attachments):
    try:
        async with QUEUE.lock(job_id):
            STORE.set_running(job_id)
            reply = await cf.generate(history, prompt, attachments)
            STORE.set_done(job_id, {"reply": reply})
    except Exception as e:
        STORE.set_error(job_id, str(e))


@app.get("/api/status", response_model=StatusResponse)
async def status():
    try:
        h = await cf.health()
        ready = bool(h.get("ready"))
    except Exception:
        ready = False
    return StatusResponse(
        clinfusion_ready=ready,
        busy=QUEUE.current_job is not None,
        current_job=QUEUE.current_job,
        queue_len=QUEUE.pending,
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    if file.filename is None:
        raise HTTPException(status_code=400, detail="missing filename")
    try:
        validate_filename(file.filename)
    except UploadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    size = getattr(file, "size", None)
    if size is not None and size > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="file too large")
    data = await file.read()
    try:
        meta = save_upload(file.filename, data)
    except UploadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    n = None
    if meta["kind"] == "nifti":
        n = nifti.n_slices(meta["path"])
    return UploadResponse(upload_id=meta["upload_id"], kind=meta["kind"],
                          filename=meta["filename"], n_slices=n)


@app.get("/api/nifti/{upload_id}/slice")
def slice_(upload_id: str, idx: int = 0, wc: float | None = None, ww: float | None = None):
    try:
        path = resolve(upload_id)
    except UploadError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not str(path).lower().endswith(config.ALLOWED_NIFTI):
        raise HTTPException(status_code=400, detail="not a NIfTI file")
    try:
        total = nifti.n_slices(path)
        png = nifti.slice_png(path, idx, wc=wc, ww=ww)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(content=png, media_type="image/png",
                    headers={"X-Total-Slices": str(total)})


@app.post("/api/radar", response_model=JobIdResponse)
async def radar_start(req: RadarRequest):
    try:
        path = resolve(req.upload_id)
    except UploadError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not str(path).lower().endswith(config.ALLOWED_NIFTI):
        raise HTTPException(status_code=400, detail="not a NIfTI file")
    jid = STORE.create("radar")
    _spawn(_radar_task(jid, path))
    return JobIdResponse(job_id=jid)


@app.get("/api/radar/{job_id}")
async def radar_get(job_id: str):
    job = STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job yok")
    result = job.get("result") or {}
    return {
        "status": job["status"],
        "progress": None,
        "findings": result.get("findings"),
        "csv_url": f"/api/radar/{job_id}/csv" if job["status"] == "done" else None,
        "error": job["error"],
    }


@app.get("/api/radar/{job_id}/csv")
async def radar_csv(job_id: str):
    csv_path = config.JOBS / job_id / "out" / f"RADAR_infer_results_{job_id}.csv"
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail="csv yok")
    return FileResponse(csv_path, filename=csv_path.name, media_type="text/csv")


@app.post("/api/chat", response_model=JobIdResponse)
async def chat_start(req: ChatRequest):
    attachments = []
    for uid in req.attachment_ids:
        try:
            attachments.append(resolve(uid))
        except UploadError as e:
            raise HTTPException(status_code=404, detail=str(e))
    jid = STORE.create("chat")
    _spawn(_chat_task(jid, req.history, req.prompt, attachments))
    return JobIdResponse(job_id=jid)


@app.get("/api/chat/{job_id}")
async def chat_get(job_id: str):
    job = STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job yok")
    result = job.get("result") or {}
    return {"status": job["status"], "reply": result.get("reply"), "error": job["error"]}


@app.post("/api/chat/reset")
async def chat_reset():
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(config.FRONTEND), html=True, check_dir=False), name="static")
