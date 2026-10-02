import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[2] / "worker"
sys.path.insert(0, str(WORKER))
from clinfusion_worker import (  # noqa: E402
    STATE,
    GenerateRequest,
    build_chat_messages,
    build_processor_kwargs,
    generate,
)


def test_build_chat_messages_text_only():
    msgs = build_chat_messages([{"role": "user", "text": "previous"},
                                {"role": "assistant", "text": "answer"}],
                               prompt="yeni", n_images=0)
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"][-1]["text"] == "yeni"
    assert msgs[-1]["content"][-1]["type"] == "text"


def test_build_chat_messages_with_images():
    msgs = build_chat_messages([], prompt="bak", n_images=2)
    types = [c["type"] for c in msgs[-1]["content"]]
    assert types == ["image", "image", "text"]


def test_build_processor_kwargs_text_only():
    kwargs = build_processor_kwargs("hi", [], None)
    assert kwargs == {"text": ["hi"], "padding": True, "return_tensors": "pt"}
    assert "images" not in kwargs
    assert "images_kwargs" not in kwargs


def test_build_processor_kwargs_with_volume():
    kwargs = build_processor_kwargs("hi", ["img"], "vol")
    assert kwargs["images"] == [["img"]]
    assert kwargs["images_kwargs"] == {"volume": ["vol"]}


def test_build_processor_kwargs_images_no_volume():
    kwargs = build_processor_kwargs("hi", ["img"], None)
    assert kwargs["images"] == [["img"]]
    assert "images_kwargs" not in kwargs


@pytest.fixture
def restore_state():
    ready, error = STATE.ready, STATE.error
    yield
    STATE.ready, STATE.error = ready, error


def test_generate_loading_returns_503(restore_state):
    STATE.ready = False
    STATE.error = None
    resp = generate(GenerateRequest(history=[], prompt="x", attachment_paths=[]))
    assert resp.status_code == 503
    assert "loading" in resp.body.decode()


def test_generate_error_returns_500(restore_state):
    STATE.ready = False
    STATE.error = "boom"
    resp = generate(GenerateRequest(history=[], prompt="x", attachment_paths=[]))
    assert resp.status_code == 500
    assert "boom" in resp.body.decode()
