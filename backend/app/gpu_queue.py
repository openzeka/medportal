import asyncio
from contextlib import asynccontextmanager
from typing import Optional


class GpuQueue:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.current_job: Optional[str] = None
        self.pending: int = 0

    @asynccontextmanager
    async def lock(self, job_id: str):
        self.pending += 1
        try:
            async with self._lock:
                self.current_job = job_id
                try:
                    yield
                finally:
                    self.current_job = None
        finally:
            self.pending -= 1


QUEUE = GpuQueue()
