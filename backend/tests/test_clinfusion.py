import httpx
import pytest
from app import clinfusion
from app.schemas import HistoryTurn


async def test_generate_ok(monkeypatch):
    class FakeResp:
        status_code = 200
        def json(self): return {"reply": "hello"}
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, json):
            assert url.endswith("/generate")
            assert json["prompt"] == "hi"
            assert json["history"] == [{"role": "user", "text": "past"}]
            assert json["attachment_paths"] == ["/tmp/a.png"]
            return FakeResp()
    monkeypatch.setattr(clinfusion.httpx, "AsyncClient", lambda **k: FakeClient())
    out = await clinfusion.generate(
        history=[HistoryTurn(role="user", text="past")],
        prompt="hi",
        attachment_paths=["/tmp/a.png"],
    )
    assert out == "hello"


async def test_generate_ok_dict_history(monkeypatch):
    class FakeResp:
        status_code = 200
        def json(self): return {"reply": "ok"}
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, json):
            assert json["history"] == [{"role": "assistant", "text": "answer"}]
            return FakeResp()
    monkeypatch.setattr(clinfusion.httpx, "AsyncClient", lambda **k: FakeClient())
    out = await clinfusion.generate(
        history=[{"role": "assistant", "text": "answer"}],
        prompt="x",
        attachment_paths=[],
    )
    assert out == "ok"


async def test_generate_timeout(monkeypatch):
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, json): raise httpx.TimeoutException("boom")
    monkeypatch.setattr(clinfusion.httpx, "AsyncClient", lambda **k: FakeClient())
    with pytest.raises(clinfusion.ClinFusionError, match="timeout"):
        await clinfusion.generate(history=[], prompt="x", attachment_paths=[])


async def test_generate_http_error(monkeypatch):
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, json): raise httpx.ConnectError("x")
    monkeypatch.setattr(clinfusion.httpx, "AsyncClient", lambda **k: FakeClient())
    with pytest.raises(clinfusion.ClinFusionError, match="error"):
        await clinfusion.generate(history=[], prompt="x", attachment_paths=[])


async def test_generate_http_status_detail(monkeypatch):
    class FakeResp:
        status_code = 500
        def json(self): return {"error": "boom"}
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, json): return FakeResp()
    monkeypatch.setattr(clinfusion.httpx, "AsyncClient", lambda **k: FakeClient())
    with pytest.raises(clinfusion.ClinFusionError, match="boom"):
        await clinfusion.generate(history=[], prompt="x", attachment_paths=[])


async def test_generate_http_status_without_json(monkeypatch):
    class FakeResp:
        status_code = 502
        def json(self): raise ValueError("no body")
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, json): return FakeResp()
    monkeypatch.setattr(clinfusion.httpx, "AsyncClient", lambda **k: FakeClient())
    with pytest.raises(clinfusion.ClinFusionError, match="502"):
        await clinfusion.generate(history=[], prompt="x", attachment_paths=[])


async def test_generate_worker_error(monkeypatch):
    class FakeResp:
        status_code = 200
        def json(self): return {"error": "model is loading"}
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, json): return FakeResp()
    monkeypatch.setattr(clinfusion.httpx, "AsyncClient", lambda **k: FakeClient())
    with pytest.raises(clinfusion.ClinFusionError, match="model is loading"):
        await clinfusion.generate(history=[], prompt="x", attachment_paths=[])


async def test_generate_non_json(monkeypatch):
    class FakeResp:
        status_code = 200
        def json(self): raise ValueError("bad json")
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, json): return FakeResp()
    monkeypatch.setattr(clinfusion.httpx, "AsyncClient", lambda **k: FakeClient())
    with pytest.raises(clinfusion.ClinFusionError, match="not JSON"):
        await clinfusion.generate(history=[], prompt="x", attachment_paths=[])


async def test_generate_non_dict(monkeypatch):
    class FakeResp:
        status_code = 200
        def json(self): return ["nope"]
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, json): return FakeResp()
    monkeypatch.setattr(clinfusion.httpx, "AsyncClient", lambda **k: FakeClient())
    with pytest.raises(clinfusion.ClinFusionError, match="unexpected"):
        await clinfusion.generate(history=[], prompt="x", attachment_paths=[])


async def test_generate_non_str_reply(monkeypatch):
    class FakeResp:
        status_code = 200
        def json(self): return {"reply": 5}
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, json): return FakeResp()
    monkeypatch.setattr(clinfusion.httpx, "AsyncClient", lambda **k: FakeClient())
    with pytest.raises(clinfusion.ClinFusionError, match="not text"):
        await clinfusion.generate(history=[], prompt="x", attachment_paths=[])


async def test_health(monkeypatch):
    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"ready": True, "loading": False}
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, url): return FakeResp()
    monkeypatch.setattr(clinfusion.httpx, "AsyncClient", lambda **k: FakeClient())
    assert (await clinfusion.health())["ready"] is True
