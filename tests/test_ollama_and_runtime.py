"""The Ollama HTTP client's failure handling, and starting the whole app."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.errors import OllamaError
from app.intelligence.ollama_client import OllamaClient


class StubHTTP:
    """Replaces httpx.Client with scripted responses."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, **kwargs):
        self.calls.append((url, {}))
        return self.handler(url, None)

    def post(self, url, json=None, **kwargs):
        self.calls.append((url, json or {}))
        return self.handler(url, json)


def response(payload, status: int = 200) -> httpx.Response:
    request = httpx.Request("GET", "http://127.0.0.1:11434/")
    return httpx.Response(status, json=payload, request=request)


@pytest.fixture
def client() -> OllamaClient:
    return OllamaClient(host="http://127.0.0.1:11434", model="qwen2.5:7b-instruct")


def install(monkeypatch, handler) -> StubHTTP:
    stub = StubHTTP(handler)
    monkeypatch.setattr("app.intelligence.ollama_client.httpx.Client", stub)
    monkeypatch.setattr("app.intelligence.ollama_client.time.sleep", lambda _s: None)
    return stub


# --- health -----------------------------------------------------------------

def test_health_is_ok_when_the_model_is_pulled(monkeypatch, client):
    install(monkeypatch, lambda url, body: response(
        {"models": [{"name": "qwen2.5:7b-instruct"}]}))
    status = client.health()
    assert status["ok"] is True
    assert status["error"] == ""


def test_health_accepts_a_model_named_without_its_tag(monkeypatch):
    client = OllamaClient(model="qwen2.5")
    install(monkeypatch, lambda url, body: response(
        {"models": [{"name": "qwen2.5:7b-instruct"}]}))
    assert client.health()["ok"] is True


def test_health_says_which_model_to_pull(monkeypatch, client):
    install(monkeypatch, lambda url, body: response({"models": [{"name": "other"}]}))
    status = client.health()
    assert status["ok"] is False
    assert "ollama pull qwen2.5:7b-instruct" in status["remedy"]


def test_health_says_how_to_start_ollama_when_it_is_down(monkeypatch, client):
    def refuse(url, body):
        raise httpx.ConnectError("connection refused")

    install(monkeypatch, refuse)
    status = client.health()
    assert status["ok"] is False
    assert "ollama serve" in status["remedy"]


def test_health_handles_a_non_ollama_service_on_the_port(monkeypatch, client):
    def not_json(url, body):
        request = httpx.Request("GET", url)
        return httpx.Response(200, text="<html>hello</html>", request=request)

    install(monkeypatch, not_json)
    assert client.health()["ok"] is False


def test_ensure_ready_raises_with_the_remedy(monkeypatch, client):
    install(monkeypatch, lambda url, body: response({"models": []}))
    with pytest.raises(OllamaError) as excinfo:
        client.ensure_ready()
    assert "ollama pull" in excinfo.value.remedy


# --- generation -------------------------------------------------------------

def test_chat_returns_the_reply(monkeypatch, client):
    stub = install(monkeypatch, lambda url, body: response(
        {"message": {"content": '{"summary": "ok"}'}}))
    assert client.chat_json("system", "user") == {"summary": "ok"}

    _url, payload = stub.calls[-1]
    assert payload["format"] == "json"
    assert payload["stream"] is False
    assert payload["options"]["num_ctx"] == 8192
    assert payload["messages"][0]["role"] == "system"


def test_max_tokens_is_passed_through(monkeypatch, client):
    stub = install(monkeypatch, lambda url, body: response(
        {"message": {"content": "{}"}}))
    client.chat("s", "u", max_tokens=500)
    assert stub.calls[-1][1]["options"]["num_predict"] == 500


def test_a_timeout_is_retried_then_reported(monkeypatch, client):
    attempts = {"count": 0}

    def timeout(url, body):
        attempts["count"] += 1
        raise httpx.ReadTimeout("too slow")

    install(monkeypatch, timeout)
    with pytest.raises(OllamaError) as excinfo:
        client.chat("s", "u")
    assert attempts["count"] == 3
    assert "ollama serve" in excinfo.value.remedy


def test_a_transient_failure_is_retried_and_then_succeeds(monkeypatch, client):
    attempts = {"count": 0}

    def flaky(url, body):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise httpx.ReadTimeout("too slow")
        return response({"message": {"content": '{"ok": true}'}})

    install(monkeypatch, flaky)
    assert client.chat_json("s", "u") == {"ok": True}
    assert attempts["count"] == 3


def test_a_bad_request_is_not_retried(monkeypatch, client):
    """A 404 will fail identically every time; retrying just wastes a minute."""
    attempts = {"count": 0}

    def not_found(url, body):
        attempts["count"] += 1
        return response({"error": "model not found"}, status=404)

    install(monkeypatch, not_found)
    with pytest.raises(OllamaError) as excinfo:
        client.chat("s", "u")
    assert attempts["count"] == 1
    assert "ollama pull" in excinfo.value.remedy


def test_an_empty_reply_is_retried_then_reported(monkeypatch, client):
    install(monkeypatch, lambda url, body: response({"message": {"content": "   "}}))
    with pytest.raises(OllamaError):
        client.chat("s", "u")


def test_unparseable_json_raises_with_a_model_suggestion(monkeypatch, client):
    install(monkeypatch, lambda url, body: response(
        {"message": {"content": "I would rather not."}}))
    with pytest.raises(OllamaError) as excinfo:
        client.chat_json("s", "u")
    assert "qwen2.5" in excinfo.value.remedy


# --- the runtime ------------------------------------------------------------

def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def runtime(settings, monkeypatch):
    from app.runtime import AppRuntime

    object.__setattr__(settings, "web_port", _free_port())
    instance = AppRuntime(settings)
    yield instance
    instance.stop()


def test_the_runtime_serves_the_dashboard(runtime):
    runtime.start(with_monitor=False)
    url = runtime.authorised_url()
    assert "/auth?token=" in url

    with httpx.Client(timeout=10.0) as client:
        assert client.get(f"{runtime.dashboard_url}api/status").status_code == 401
        authed = client.get(url, follow_redirects=True)
        assert authed.status_code == 200
        assert "Teams Notes" in authed.text
        assert client.get(f"{runtime.dashboard_url}api/status").status_code == 200


def test_the_runtime_stops_cleanly_and_can_be_stopped_twice(runtime):
    runtime.start(with_monitor=False)
    runtime.stop()
    runtime.stop()
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"{runtime.dashboard_url}api/status", timeout=3.0)


def test_a_busy_port_is_reported_clearly(settings, monkeypatch):
    """The common first-run problem: something else already has the port."""
    import socket

    from app.runtime import AppRuntime

    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    object.__setattr__(settings, "web_port", blocker.getsockname()[1])

    instance = AppRuntime(settings)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            instance.start(with_monitor=False)
        assert "WEB_PORT" in str(excinfo.value)
    finally:
        instance.stop()
        blocker.close()
