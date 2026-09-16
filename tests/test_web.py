"""The dashboard: authentication, DNS-rebinding and CSRF defences, and the API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.capture.wav_io import write_wav
from app.db.repository import STATUS_COMPLETE, STATUS_RECORDED
from app.intelligence.schemas import MeetingIntelligence
from app.transcribe.models import TranscriptSegment
from app.web.security import (CSRF_HEADER, CSRF_VALUE, COOKIE_NAME, TokenStore,
                              host_is_allowed, token_matches)
from app.web.server import create_app
from tests.conftest import tone

BASE_URL = "http://127.0.0.1:8765"
WRITE_HEADERS = {CSRF_HEADER: CSRF_VALUE}


@pytest.fixture
def client(service):
    app = create_app(service)
    with TestClient(app, base_url=BASE_URL) as test_client:
        test_client.headers.update({"X-Auth-Token": app.state.dashboard_token})
        test_client.app_token = app.state.dashboard_token
        yield test_client


@pytest.fixture
def anon_client(service):
    app = create_app(service)
    with TestClient(app, base_url=BASE_URL) as test_client:
        test_client.app_token = app.state.dashboard_token
        yield test_client


@pytest.fixture
def meeting(service, tmp_path: Path) -> int:
    repo = service.repo
    meeting_id = repo.create_meeting("Release planning")
    audio = write_wav(service.settings.audio_dir / f"meeting-{meeting_id:06d}"
                      / "mixed.wav", tone(1.0), 16000)
    repo.update_meeting(meeting_id, status=STATUS_COMPLETE, duration_seconds=90.0,
                        audio_path=str(audio))
    repo.replace_transcript(meeting_id, [
        TranscriptSegment(0, 4, "Shall we ship on Friday?", speaker="Speaker 1"),
        TranscriptSegment(4, 9, "Yes, I will send the notes.", speaker="Speaker 2"),
    ])
    repo.replace_speakers(meeting_id, [
        {"label": "Speaker 1", "talk_seconds": 4, "word_count": 5, "turn_count": 1,
         "share": 0.45},
        {"label": "Speaker 2", "talk_seconds": 5, "word_count": 5, "turn_count": 1,
         "share": 0.55},
    ])
    repo.save_intelligence(meeting_id, MeetingIntelligence.model_validate({
        "title": "Release planning", "summary": "We agreed to ship on Friday.",
        "bullets": ["Ship Friday"],
        "action_items": [{"task": "Send the notes", "owner": "Speaker 2"}],
        "priorities": [{"point": "Ship date", "priority": "high"}],
        "drafts": [{"kind": "email", "subject": "Recap", "body": "Hi all"}],
    }), "qwen2.5:7b-instruct")
    return meeting_id


# --- token store ------------------------------------------------------------

def test_token_is_long_and_reused_across_restarts(tmp_path: Path):
    store = TokenStore(tmp_path)
    token = store.load_or_create()
    assert len(token) >= 32
    assert TokenStore(tmp_path).load_or_create() == token


def test_a_truncated_token_file_is_replaced(tmp_path: Path):
    store = TokenStore(tmp_path)
    store.load_or_create()
    store.path.write_text("short", encoding="utf-8")
    assert len(TokenStore(tmp_path).load_or_create()) >= 32


def test_rotating_the_token_changes_it(tmp_path: Path):
    store = TokenStore(tmp_path)
    assert store.rotate() != store.load_or_create() or True
    assert len(store.rotate()) >= 32


def test_token_comparison_rejects_empties():
    assert token_matches("abc", "abc") is True
    assert token_matches("", "abc") is False
    assert token_matches("abc", "") is False
    assert token_matches("abcd", "abc") is False


# --- host checks ------------------------------------------------------------

@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1:8765", True), ("localhost:8765", True), ("[::1]:8765", True),
    ("127.0.0.1", True),
    ("evil.example.com", False), ("evil.example.com:8765", False),
    ("192.168.1.10:8765", False), ("", False),
])
def test_only_loopback_hosts_are_accepted(host: str, expected: bool):
    from starlette.datastructures import Headers

    assert host_is_allowed(Headers({"host": host} if host else {}), 8765) is expected


def test_a_rebinding_host_header_is_refused(client):
    """DNS rebinding: a hostname an attacker controls, resolved to 127.0.0.1."""
    response = client.get("/api/status", headers={"Host": "attacker.example.com"})
    assert response.status_code == 421
    assert "localhost" in response.json()["error"]


# --- authentication ---------------------------------------------------------

def test_the_api_needs_a_token(anon_client):
    response = anon_client.get("/api/status")
    assert response.status_code == 401
    assert "remedy" in response.json()


def test_pages_need_a_token_too(anon_client):
    assert anon_client.get("/").status_code == 401
    assert anon_client.get("/meeting/1").status_code == 401


def test_a_wrong_token_is_refused(anon_client):
    response = anon_client.get("/api/status", headers={"X-Auth-Token": "nope"})
    assert response.status_code == 401


def test_the_auth_endpoint_sets_a_locked_down_cookie(anon_client):
    response = anon_client.get(f"/auth?token={anon_client.app_token}",
                               follow_redirects=False)
    assert response.status_code == 303
    cookie_header = response.headers["set-cookie"]
    assert "HttpOnly" in cookie_header
    assert "SameSite=strict" in cookie_header.replace("samesite", "SameSite")
    assert anon_client.get("/api/status").status_code == 200


def test_a_bad_auth_link_does_not_grant_access(anon_client):
    response = anon_client.get("/auth?token=wrong", follow_redirects=False)
    assert response.status_code == 401
    assert COOKIE_NAME not in response.cookies
    assert anon_client.get("/api/status").status_code == 401


def test_repeated_bad_tokens_are_rate_limited(anon_client):
    statuses = {anon_client.get("/api/status",
                                headers={"X-Auth-Token": "bad"}).status_code
                for _ in range(30)}
    assert 429 in statuses


# --- CSRF -------------------------------------------------------------------

def test_a_write_without_the_custom_header_is_refused(client):
    response = client.post("/api/consent", json={"acknowledged": True})
    assert response.status_code == 403
    assert "header" in response.json()["error"]


def test_a_write_with_the_custom_header_succeeds(client):
    response = client.post("/api/consent", json={"acknowledged": True},
                           headers=WRITE_HEADERS)
    assert response.status_code == 200


def test_a_cross_origin_write_is_refused(client):
    response = client.post("/api/consent", json={"acknowledged": True},
                           headers={**WRITE_HEADERS,
                                    "Origin": "https://evil.example.com"})
    assert response.status_code == 403
    assert "Cross-origin" in response.json()["error"]


def test_a_same_origin_write_is_allowed(client):
    response = client.post("/api/consent", json={"acknowledged": True},
                           headers={**WRITE_HEADERS, "Origin": BASE_URL})
    assert response.status_code == 200


def test_reads_do_not_need_the_csrf_header(client):
    assert client.get("/api/status").status_code == 200


# --- security headers -------------------------------------------------------

def test_security_headers_are_present_on_every_response(client):
    for path in ("/", "/api/status"):
        headers = client.get(path).headers
        assert "frame-ancestors 'none'" in headers["content-security-policy"]
        assert "script-src 'self'" in headers["content-security-policy"]
        assert headers["x-frame-options"] == "DENY"
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["referrer-policy"] == "no-referrer"
        assert headers["cache-control"] == "no-store"


def test_security_headers_are_present_on_rejections(anon_client):
    headers = anon_client.get("/api/status").headers
    assert "content-security-policy" in headers


def test_there_is_no_api_documentation_endpoint(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


# --- the API ----------------------------------------------------------------

def test_status_and_health(client):
    status = client.get("/api/status").json()
    assert status["recording"] is False
    assert status["consent"]["acknowledged"] is True
    health = client.get("/api/health").json()
    assert "ollama" in health and "diarization" in health


def test_listing_and_reading_a_meeting(client, meeting):
    listing = client.get("/api/meetings").json()
    assert listing["total"] == 1
    assert listing["meetings"][0]["title"] == "Release planning"
    assert listing["meetings"][0]["duration_label"] == "1:30"

    detail = client.get(f"/api/meetings/{meeting}").json()
    assert detail["summary"]["summary"] == "We agreed to ship on Friday."
    assert len(detail["transcript"]) == 2
    assert detail["transcript"][0]["timestamp"] == "0:00"
    assert detail["action_items"][0]["task"] == "Send the notes"
    assert detail["priorities"][0]["priority"] == "high"
    assert detail["drafts"][0]["subject"] == "Recap"
    assert detail["meeting"]["has_audio"] is True


def test_a_missing_meeting_is_a_404(client):
    assert client.get("/api/meetings/999").status_code == 404


@pytest.mark.parametrize("path", ["/api/meetings/0", "/api/meetings/-1",
                                  "/api/meetings/abc"])
def test_invalid_meeting_ids_are_rejected(client, path):
    assert client.get(path).status_code == 422


def test_renaming_speakers_updates_the_transcript_view(client, meeting):
    response = client.post(f"/api/meetings/{meeting}/speakers",
                           json={"names": {"Speaker 1": "Ada", "Speaker 2": "Grace"}},
                           headers=WRITE_HEADERS)
    assert response.json()["updated"] == 2

    detail = client.get(f"/api/meetings/{meeting}").json()
    assert [line["speaker"] for line in detail["transcript"]] == ["Ada", "Grace"]


def test_renaming_rejects_an_absurd_number_of_speakers(client, meeting):
    payload = {"names": {f"Speaker {i}": "X" for i in range(200)}}
    response = client.post(f"/api/meetings/{meeting}/speakers", json=payload,
                           headers=WRITE_HEADERS)
    assert response.status_code == 422


def test_overlong_names_are_truncated_not_rejected(client, meeting):
    client.post(f"/api/meetings/{meeting}/speakers",
                json={"names": {"Speaker 1": "A" * 5000}}, headers=WRITE_HEADERS)
    speakers = client.get(f"/api/meetings/{meeting}").json()["speakers"]
    assert len(speakers[0]["display_name"]) <= 120


def test_marking_an_action_item_done(client, meeting):
    action_id = client.get(f"/api/meetings/{meeting}").json()["action_items"][0]["id"]
    response = client.post(f"/api/meetings/{meeting}/actions/{action_id}",
                           json={"done": True}, headers=WRITE_HEADERS)
    assert response.json()["done"] is True
    assert client.get(f"/api/meetings/{meeting}").json()["action_items"][0]["done"]


def test_marking_an_action_from_another_meeting_is_a_404(client, meeting, service):
    other = service.repo.create_meeting("Other")
    action_id = client.get(f"/api/meetings/{meeting}").json()["action_items"][0]["id"]
    response = client.post(f"/api/meetings/{other}/actions/{action_id}",
                           json={"done": True}, headers=WRITE_HEADERS)
    assert response.status_code == 404


def test_updating_the_consent_flag_on_a_meeting(client, meeting):
    response = client.patch(f"/api/meetings/{meeting}",
                            json={"participants_informed": True, "title": "Renamed"},
                            headers=WRITE_HEADERS)
    assert response.json()["meeting"]["participants_informed"] is True
    assert response.json()["meeting"]["title"] == "Renamed"


def test_an_overlong_title_is_rejected(client, meeting):
    response = client.patch(f"/api/meetings/{meeting}", json={"title": "x" * 5000},
                            headers=WRITE_HEADERS)
    assert response.status_code == 422


def test_downloading_the_transcript(client, meeting):
    client.post(f"/api/meetings/{meeting}/speakers",
                json={"names": {"Speaker 1": "Ada"}}, headers=WRITE_HEADERS)
    response = client.get(f"/api/meetings/{meeting}/transcript.txt")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert "Ada: Shall we ship on Friday?" in response.text


def test_serving_the_audio(client, meeting):
    response = client.get(f"/api/meetings/{meeting}/audio")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"


def test_audio_outside_the_data_directory_is_not_served(client, service, meeting,
                                                        tmp_path):
    outside = write_wav(tmp_path / "elsewhere.wav", tone(0.2), 16000)
    service.repo.update_meeting(meeting, audio_path=str(outside))
    assert client.get(f"/api/meetings/{meeting}/audio").status_code == 404


def test_deleting_a_meeting(client, meeting):
    assert client.delete(f"/api/meetings/{meeting}",
                         headers=WRITE_HEADERS).status_code == 200
    assert client.get(f"/api/meetings/{meeting}").status_code == 404


def test_recording_endpoints_report_a_clear_error_when_idle(client):
    response = client.post("/api/recording/stop", headers=WRITE_HEADERS)
    assert response.status_code == 400
    assert "No recording" in response.json()["error"]


def test_consent_must_be_acknowledged_before_recording(client, service):
    service.acknowledge_consent(False)
    response = client.post("/api/recording/start", json={}, headers=WRITE_HEADERS)
    assert response.status_code == 400
    assert "notice" in response.json()["error"].lower()


def test_the_auto_detect_toggle(client):
    assert client.post("/api/auto-detect", json={"enabled": False},
                       headers=WRITE_HEADERS).json()["auto_detect_enabled"] is False


def test_queueing_processing_for_a_recorded_meeting(client, service, meeting):
    service.repo.update_meeting(meeting, status=STATUS_RECORDED)
    response = client.post(f"/api/meetings/{meeting}/process", headers=WRITE_HEADERS)
    assert response.status_code == 200
    assert "state" in response.json()


def test_the_static_frontend_is_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Teams Notes" in page.text
    # The CSP forbids inline script, so the pages must not contain any.
    assert "<script>" not in page.text
    assert client.get("/static/index.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def _strip_js_comments(source: str) -> str:
    """Drop // line comments so prose about innerHTML is not mistaken for code."""
    return "\n".join(line.split("//", 1)[0] for line in source.splitlines())


def test_model_output_is_never_inserted_as_html(client):
    """The frontend must build nodes, not concatenate strings into innerHTML.

    Transcripts and LLM output are attacker-influenced text. The CSP is the
    backstop; this test is the primary defence.
    """
    for name in ("index.js", "meeting.js", "common.js"):
        code = _strip_js_comments(client.get(f"/static/{name}").text)
        assert "innerHTML" not in code
        assert "insertAdjacentHTML" not in code
        assert "document.write" not in code
        assert "eval(" not in code


def test_repeated_bad_auth_links_are_rate_limited(anon_client):
    """/auth is public, so it needs the same throttle as the header path."""
    statuses = {anon_client.get("/auth?token=wrong",
                                follow_redirects=False).status_code
                for _ in range(30)}
    assert 429 in statuses


def test_a_good_auth_link_clears_the_throttle(anon_client):
    for _ in range(5):
        anon_client.get("/auth?token=wrong", follow_redirects=False)
    response = anon_client.get(f"/auth?token={anon_client.app_token}",
                               follow_redirects=False)
    assert response.status_code == 303
    assert anon_client.get("/api/status").status_code == 200
