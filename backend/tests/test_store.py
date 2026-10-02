from app.store import JobStore


def test_create_and_update():
    s = JobStore()
    jid = s.create("radar")
    assert s.get(jid)["status"] == "queued"
    s.set_running(jid)
    assert s.get(jid)["status"] == "running"
    s.set_done(jid, {"findings": []})
    assert s.get(jid)["status"] == "done"
    assert s.get(jid)["result"] == {"findings": []}


def test_error():
    s = JobStore()
    jid = s.create("chat")
    s.set_error(jid, "boom")
    assert s.get(jid)["status"] == "error"
    assert "boom" in s.get(jid)["error"]
