"""The FastAPI app behind the dashboard.

Routes fall into three groups: a couple of HTML pages, a JSON API the frontend
drives, and one audio endpoint. Every response goes through SecurityMiddleware
(see security.py), and every request body is validated by a pydantic model with
explicit length limits, so nothing unbounded reaches the database.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, HTTPException, Path as PathParam, Query
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, Response)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.diarize.merge import apply_speaker_names
from app.errors import NotFoundError, TeamsNotesError
from app.logging_setup import get_logger
from app.pipeline.service import MeetingService
from app.transcribe.models import format_timestamp, render_transcript
from app.web.security import (COOKIE_NAME, FailureLimiter, SecurityMiddleware,
                              TokenStore, apply_security_headers, token_matches)

log = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_NAME_LENGTH = 120
MAX_TITLE_LENGTH = 200


# --- request models ---------------------------------------------------------

class StartRecordingRequest(BaseModel):
    title: str = Field(default="", max_length=MAX_TITLE_LENGTH)
    participants_informed: bool = False


class RenameSpeakersRequest(BaseModel):
    names: dict[str, str] = Field(default_factory=dict)
    regenerate: bool = False

    @field_validator("names")
    @classmethod
    def _limit(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 50:
            raise ValueError("too many speakers")
        return {
            str(label)[:MAX_NAME_LENGTH]: str(name).strip()[:MAX_NAME_LENGTH]
            for label, name in value.items()
        }


class MeetingUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=MAX_TITLE_LENGTH)
    participants_informed: bool | None = None
    consent_note: str | None = Field(default=None, max_length=1000)


class ConsentRequest(BaseModel):
    acknowledged: bool | None = None
    reminder_enabled: bool | None = None


class AutoDetectRequest(BaseModel):
    enabled: bool


class ActionDoneRequest(BaseModel):
    done: bool


# --- app --------------------------------------------------------------------

def create_app(service: MeetingService) -> FastAPI:
    """Build the FastAPI app around an already-started MeetingService."""
    settings = service.settings
    tokens = TokenStore(settings.runtime_dir)
    token = tokens.load_or_create()

    app = FastAPI(
        title="Teams Notes",
        version="1.0.0",
        docs_url=None,       # no interactive docs: fewer endpoints, less surface
        redoc_url=None,
        openapi_url=None,
    )
    app.state.service = service
    app.state.tokens = tokens
    app.state.dashboard_token = token

    # One throttle shared by the middleware and the /auth route below.
    limiter = FailureLimiter()
    app.state.limiter = limiter
    app.add_middleware(SecurityMiddleware, token_store=tokens,
                       port=settings.web_port, limiter=limiter)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def get_service() -> MeetingService:
        return service

    # --- error handling ---------------------------------------------------

    @app.exception_handler(TeamsNotesError)
    async def handle_app_error(_request, exc: TeamsNotesError) -> JSONResponse:
        status = 404 if isinstance(exc, NotFoundError) else 400
        return apply_security_headers(JSONResponse(
            {"error": exc.message, "remedy": exc.remedy}, status_code=status))

    # --- pages -------------------------------------------------------------

    @app.get("/auth", include_in_schema=False)
    async def auth(token_param: str = Query(default="", alias="token")):
        """Exchange the startup token for a session cookie, then redirect."""
        if not token_matches(token_param, app.state.dashboard_token):
            limiter.record_failure()
            return apply_security_headers(HTMLResponse(
                "<!doctype html><meta charset=utf-8><title>Teams Notes</title>"
                "<p>That link is not valid. Open the dashboard from the tray icon, "
                "or run <code>python main.py dashboard</code>.</p>",
                status_code=401))

        limiter.reset()
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(
            COOKIE_NAME, app.state.dashboard_token,
            httponly=True,      # not readable from JavaScript
            samesite="strict",  # never sent on a cross-site request
            secure=False,       # loopback HTTP; there is no TLS to require
            max_age=60 * 60 * 24 * 30,
            path="/",
        )
        return apply_security_headers(response)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index():
        return _page("index.html")

    @app.get("/meeting/{meeting_id}", response_class=HTMLResponse,
             include_in_schema=False)
    async def meeting_page(meeting_id: int = PathParam(ge=1)):
        return _page("meeting.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        icon = STATIC_DIR / "favicon.svg"
        if icon.is_file():
            return apply_security_headers(FileResponse(icon, media_type="image/svg+xml"))
        return apply_security_headers(Response(status_code=204))

    # --- status ------------------------------------------------------------

    @app.get("/api/status")
    async def status(svc: MeetingService = Depends(get_service)):
        return svc.status()

    @app.get("/api/health")
    async def health(svc: MeetingService = Depends(get_service)):
        return svc.health()

    # --- consent -----------------------------------------------------------

    @app.get("/api/consent")
    async def get_consent(svc: MeetingService = Depends(get_service)):
        return svc.consent_state()

    @app.post("/api/consent")
    async def set_consent(payload: ConsentRequest = Body(...),
                          svc: MeetingService = Depends(get_service)):
        if payload.acknowledged is not None:
            svc.acknowledge_consent(payload.acknowledged)
        if payload.reminder_enabled is not None:
            svc.set_consent_reminder(payload.reminder_enabled)
        return svc.consent_state()

    @app.post("/api/auto-detect")
    async def set_auto_detect(payload: AutoDetectRequest = Body(...),
                              svc: MeetingService = Depends(get_service)):
        svc.set_auto_detect(payload.enabled)
        return {"auto_detect_enabled": svc.auto_detect_enabled}

    # --- recording ---------------------------------------------------------

    @app.post("/api/recording/start")
    async def start_recording(payload: StartRecordingRequest = Body(default=None),
                              svc: MeetingService = Depends(get_service)):
        payload = payload or StartRecordingRequest()
        meeting_id = svc.start_recording(
            title=payload.title, source="manual",
            participants_informed=payload.participants_informed)
        return {"meeting_id": meeting_id, "status": svc.status()}

    @app.post("/api/recording/stop")
    async def stop_recording(svc: MeetingService = Depends(get_service)):
        return svc.stop_recording(process=True)

    @app.post("/api/recording/discard")
    async def discard_recording(svc: MeetingService = Depends(get_service)):
        meeting_id = svc.discard_recording()
        if meeting_id is None:
            raise HTTPException(status_code=400, detail="No recording is in progress.")
        return {"meeting_id": meeting_id, "discarded": True}

    # --- meetings ----------------------------------------------------------

    @app.get("/api/meetings")
    async def list_meetings(limit: int = Query(default=50, ge=1, le=200),
                            offset: int = Query(default=0, ge=0),
                            svc: MeetingService = Depends(get_service)):
        meetings = svc.repo.list_meetings(limit=limit, offset=offset)
        return {
            "meetings": [_meeting_summary(svc, meeting) for meeting in meetings],
            "total": svc.repo.count_meetings(),
        }

    @app.get("/api/meetings/{meeting_id}")
    async def get_meeting(meeting_id: int = PathParam(ge=1),
                          svc: MeetingService = Depends(get_service)):
        return _meeting_detail(svc, meeting_id)

    @app.patch("/api/meetings/{meeting_id}")
    async def update_meeting(meeting_id: int = PathParam(ge=1),
                             payload: MeetingUpdateRequest = Body(...),
                             svc: MeetingService = Depends(get_service)):
        svc.repo.get_meeting(meeting_id)
        updates: dict[str, Any] = {}
        if payload.title is not None:
            updates["title"] = payload.title.strip()[:MAX_TITLE_LENGTH]
        if payload.participants_informed is not None:
            updates["participants_informed"] = 1 if payload.participants_informed else 0
        if payload.consent_note is not None:
            updates["consent_note"] = payload.consent_note.strip()[:1000]
        if updates:
            svc.repo.update_meeting(meeting_id, **updates)
        return _meeting_detail(svc, meeting_id)

    @app.delete("/api/meetings/{meeting_id}")
    async def delete_meeting(meeting_id: int = PathParam(ge=1),
                             svc: MeetingService = Depends(get_service)):
        svc.delete_meeting(meeting_id)
        return {"deleted": meeting_id}

    @app.post("/api/meetings/{meeting_id}/process")
    async def process_meeting(meeting_id: int = PathParam(ge=1),
                              svc: MeetingService = Depends(get_service)):
        return {"queued": svc.queue_processing(meeting_id),
                "state": svc.jobs.job_state(f"process-{meeting_id}")}

    @app.post("/api/meetings/{meeting_id}/regenerate")
    async def regenerate_meeting(meeting_id: int = PathParam(ge=1),
                                 svc: MeetingService = Depends(get_service)):
        return {"queued": svc.queue_regenerate(meeting_id),
                "state": svc.jobs.job_state(f"regenerate-{meeting_id}")}

    @app.post("/api/meetings/{meeting_id}/speakers")
    async def rename_speakers(meeting_id: int = PathParam(ge=1),
                              payload: RenameSpeakersRequest = Body(...),
                              svc: MeetingService = Depends(get_service)):
        svc.repo.get_meeting(meeting_id)
        updated = svc.repo.rename_speakers(meeting_id, payload.names)
        queued = svc.queue_regenerate(meeting_id) if payload.regenerate else False
        return {"updated": updated, "regenerating": queued,
                "speakers": svc.repo.get_speakers(meeting_id)}

    @app.post("/api/meetings/{meeting_id}/actions/{action_id}")
    async def set_action_done(meeting_id: int = PathParam(ge=1),
                              action_id: int = PathParam(ge=1),
                              payload: ActionDoneRequest = Body(...),
                              svc: MeetingService = Depends(get_service)):
        if not svc.repo.set_action_done(meeting_id, action_id, payload.done):
            raise HTTPException(status_code=404, detail="No such action item.")
        return {"id": action_id, "done": payload.done}

    @app.get("/api/meetings/{meeting_id}/transcript.txt", response_class=Response)
    async def transcript_text(meeting_id: int = PathParam(ge=1),
                              svc: MeetingService = Depends(get_service)):
        meeting = svc.repo.get_meeting(meeting_id)
        segments = apply_speaker_names(svc.repo.get_transcript(meeting_id),
                                       svc.repo.speaker_name_map(meeting_id))
        header = f"{meeting.get('title') or 'Meeting'}\n{meeting.get('started_at')}\n\n"
        return apply_security_headers(Response(
            content=header + render_transcript(segments),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition":
                     f'attachment; filename="meeting-{meeting_id}-transcript.txt"'},
        ))

    @app.get("/api/meetings/{meeting_id}/audio")
    async def meeting_audio(meeting_id: int = PathParam(ge=1),
                            svc: MeetingService = Depends(get_service)):
        path = svc.audio_path_for(meeting_id)
        if path is None:
            raise HTTPException(status_code=404,
                                detail="This meeting's audio is no longer on disk.")
        return apply_security_headers(FileResponse(path, media_type="audio/wav"))

    return app


# --- helpers ----------------------------------------------------------------

def _page(name: str) -> HTMLResponse:
    path = STATIC_DIR / name
    if not path.is_file():  # pragma: no cover - only if the install is broken
        return apply_security_headers(HTMLResponse(
            "<!doctype html><p>Dashboard files are missing.</p>", status_code=500))
    return apply_security_headers(
        HTMLResponse(path.read_text(encoding="utf-8")))


def _meeting_summary(service: MeetingService, meeting: dict) -> dict[str, Any]:
    summary = service.repo.get_summary(int(meeting["id"]))
    return {
        "id": meeting["id"],
        "title": meeting.get("title") or "Untitled meeting",
        "started_at": meeting.get("started_at"),
        "ended_at": meeting.get("ended_at"),
        "duration_seconds": meeting.get("duration_seconds", 0),
        "duration_label": format_timestamp(meeting.get("duration_seconds") or 0),
        "status": meeting.get("status"),
        "stage": meeting.get("stage", ""),
        "progress": meeting.get("progress", 0),
        "error": meeting.get("error", ""),
        "source": meeting.get("source", "manual"),
        "participants_informed": meeting.get("participants_informed", False),
        "summary": (summary or {}).get("summary", ""),
    }


def _meeting_detail(service: MeetingService, meeting_id: int) -> dict[str, Any]:
    meeting = service.repo.get_meeting(meeting_id)
    names = service.repo.speaker_name_map(meeting_id)
    segments = apply_speaker_names(service.repo.get_transcript(meeting_id), names)
    summary = service.repo.get_summary(meeting_id) or {}

    return {
        "meeting": {
            **_meeting_summary(service, meeting),
            "warnings": meeting.get("warnings", []),
            "consent_note": meeting.get("consent_note", ""),
            "has_audio": service.audio_path_for(meeting_id) is not None,
        },
        "summary": {
            "title": summary.get("title", ""),
            "summary": summary.get("summary", ""),
            "bullets": summary.get("bullets", []),
            "model": summary.get("model", ""),
        },
        "transcript": [
            {
                "start": segment.start,
                "end": segment.end,
                "timestamp": format_timestamp(segment.start),
                "speaker": segment.speaker or "Unknown",
                "text": segment.text,
            }
            for segment in segments
        ],
        "speakers": service.repo.get_speakers(meeting_id),
        "action_items": service.repo.get_action_items(meeting_id),
        "priorities": service.repo.get_priorities(meeting_id),
        "drafts": service.repo.get_drafts(meeting_id),
    }
