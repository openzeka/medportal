import threading
import uuid
import time
from typing import Any, Dict, Optional


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create(self, kind: str) -> str:
        jid = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[jid] = {
                "id": jid,
                "kind": kind,
                "status": "queued",
                "result": None,
                "error": None,
                "created": time.time(),
            }
        return jid

    def _update(self, jid: str, **kw: Any) -> None:
        with self._lock:
            if jid in self._jobs:
                self._jobs[jid].update(kw)

    def set_running(self, jid: str) -> None:
        self._update(jid, status="running")

    def set_done(self, jid: str, result: Any) -> None:
        self._update(jid, status="done", result=result)

    def set_error(self, jid: str, error: str) -> None:
        self._update(jid, status="error", error=error)

    def get(self, jid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(jid)
            return dict(job) if job else None


STORE = JobStore()
