from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    upload_id: str
    kind: str
    filename: str
    n_slices: Optional[int] = None


class RadarRequest(BaseModel):
    upload_id: str


class JobIdResponse(BaseModel):
    job_id: str


class HistoryTurn(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    prompt: str
    history: List[HistoryTurn] = []
    attachment_ids: List[str] = []


class StatusResponse(BaseModel):
    gpu: str = "GB10"
    radar_ready: bool = True
    clinfusion_ready: bool
    busy: bool
    current_job: Optional[str]
    queue_len: int


class JobResponse(BaseModel):
    id: str
    kind: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
