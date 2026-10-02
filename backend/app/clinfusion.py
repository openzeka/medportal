import httpx

from . import config


class ClinFusionError(Exception):
    pass


async def health() -> dict:
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{config.WORKER_URL}/health")
        r.raise_for_status()
        return r.json()


async def generate(history, prompt: str, attachment_paths) -> str:
    payload = {
        "history": [{"role": t.role if hasattr(t, "role") else t["role"],
                     "text": t.text if hasattr(t, "text") else t["text"]} for t in history],
        "prompt": prompt,
        "attachment_paths": [str(p) for p in attachment_paths],
    }
    try:
        async with httpx.AsyncClient(timeout=config.CHAT_TIMEOUT) as client:
            r = await client.post(f"{config.WORKER_URL}/generate", json=payload)
            if r.status_code >= 400:
                try:
                    detail = r.json().get("error")
                except Exception:
                    detail = None
                raise ClinFusionError(
                    f"ClinFusion error ({r.status_code}): {detail or 'unknown'}"
                )
            try:
                data = r.json()
            except Exception as e:
                raise ClinFusionError(f"worker response is not JSON: {e}") from e
            if not isinstance(data, dict):
                raise ClinFusionError("worker response has unexpected shape")
            if "reply" not in data:
                raise ClinFusionError(data.get("error", "invalid worker response"))
            if not isinstance(data["reply"], str):
                raise ClinFusionError("worker reply is not text")
            return data["reply"]
    except httpx.TimeoutException as e:
        raise ClinFusionError(f"ClinFusion timeout: {e}") from e
    except httpx.HTTPError as e:
        raise ClinFusionError(f"ClinFusion error: {e}") from e
