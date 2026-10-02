import asyncio

from app.gpu_queue import GpuQueue


async def test_serialized():
    q = GpuQueue()
    order = []

    async def job(n, d):
        async with q.lock("job%d" % n):
            order.append(("start", n))
            await asyncio.sleep(d)
            order.append(("end", n))

    await asyncio.gather(job(1, 0.05), job(2, 0.01))
    assert order == [("start", 1), ("end", 1), ("start", 2), ("end", 2)]


def test_current_job():
    q = GpuQueue()
    assert q.current_job is None
    q.current_job = "abc"
    assert q.current_job == "abc"
