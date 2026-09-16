#!/usr/bin/env python3
"""Regenerate the dashboard screenshots in docs/.

This starts the real app against a throwaway data directory, seeds it with
plainly fictional demo content, and photographs the actual dashboard with a
headless browser. Nothing here is a mockup: what lands in docs/ is the same
HTML, CSS and JavaScript the app serves.

Needs Playwright, which is not a project dependency:

    pip install playwright
    playwright install chromium

Then:

    python tools/make_screenshots.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.capture.wav_io import write_wav  # noqa: E402
from app.db.repository import STATUS_COMPLETE, STATUS_PROCESSING  # noqa: E402
from app.intelligence.schemas import MeetingIntelligence  # noqa: E402
from app.runtime import AppRuntime  # noqa: E402
from app.transcribe.models import TranscriptSegment  # noqa: E402
from config import load_settings  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "docs"
VIEWPORT = {"width": 1180, "height": 900}

# --- demo content -----------------------------------------------------------
# Invented people, an invented project. Nothing here is a real meeting.

TRANSCRIPT = [
    (0.0, 6.2, "Priya Raman",
     "Right, let's start with the Atlas migration. Where are we on the "
     "read-path cutover?"),
    (6.8, 17.4, "Daniel Okafor",
     "Read path is done in staging and it's been stable for four days. The "
     "write path is the one I'm worried about. There's a dual-write window "
     "where we could lose rows if the queue backs up."),
    (18.0, 24.1, "Priya Raman",
     "How long is that window realistically?"),
    (24.5, 33.8, "Daniel Okafor",
     "About ninety seconds under normal load. Under peak it could stretch to "
     "five or six minutes, which is long enough to matter."),
    (34.2, 44.0, "Sofia Lindqvist",
     "We hit exactly that on the billing migration last year. We ended up "
     "adding a reconciliation job that replays anything the queue dropped. "
     "It's not elegant but it saved us."),
    (44.6, 52.3, "Daniel Okafor",
     "That's reassuring. I can have a reconciliation job written by Wednesday "
     "if we agree that's the approach."),
    (52.9, 61.5, "Priya Raman",
     "Let's do that. Sofia, can you dig out whatever you wrote for billing so "
     "Daniel isn't starting from scratch?"),
    (62.0, 66.8, "Sofia Lindqvist",
     "Yes, I'll send it over this afternoon."),
    (67.4, 79.2, "Priya Raman",
     "Good. Second thing: we need a decision on the cutover date. Marketing "
     "wants it before the conference, which means the week of the twelfth."),
    (79.8, 91.6, "Sofia Lindqvist",
     "That's tight. Two of us are out that week and I'd rather not cut over "
     "with half a team. Can we make a case for the week after?"),
    (92.1, 101.4, "Priya Raman",
     "I think we can. I'll talk to Marcus and push for the nineteenth. If he "
     "says no, we'll revisit staffing."),
    (102.0, 108.7, "Daniel Okafor",
     "One more thing — the old cluster's certificate expires on the first of "
     "next month. That's a hard deadline whatever we decide."),
    (109.2, 116.5, "Priya Raman",
     "Noted, that's a real constraint. Let's make sure it's on the plan and "
     "not just in this conversation."),
]

NOTES = {
    "title": "Atlas migration: cutover planning",
    "summary": (
        "The team reviewed progress on the Atlas migration and agreed how to "
        "handle the risk in the dual-write window. The read path is already "
        "stable in staging; the write path needs a reconciliation job to "
        "recover anything dropped while the queue backs up, reusing the "
        "approach from last year's billing migration. The cutover date is not "
        "settled: marketing wants the week of the 12th, but the team is "
        "short-staffed that week and will make a case for the 19th. The old "
        "cluster's certificate expiry on the 1st is a hard deadline either way."
    ),
    "bullets": [
        "Read path is done and has been stable in staging for four days.",
        "The dual-write window risks losing rows: 90 seconds typically, up to "
        "six minutes at peak load.",
        "A reconciliation job that replays dropped rows was agreed as the fix, "
        "reusing the pattern from the billing migration.",
        "Marketing wants the cutover in the week of the 12th; the team is "
        "short-staffed and prefers the 19th.",
        "The old cluster's certificate expires on the 1st, which is a hard "
        "deadline regardless of the date chosen.",
    ],
    "action_items": [
        {"task": "Write the reconciliation job for the dual-write window",
         "owner": "Daniel Okafor", "due": "Wednesday", "priority": "high",
         "context": "Without it, rows can be lost whenever the queue backs up."},
        {"task": "Send over the billing migration reconciliation code",
         "owner": "Sofia Lindqvist", "due": "This afternoon", "priority": "high",
         "context": "So Daniel is not starting from scratch."},
        {"task": "Make the case to Marcus for cutting over on the 19th",
         "owner": "Priya Raman", "due": "", "priority": "medium",
         "context": "Two team members are away the week of the 12th."},
        {"task": "Add the certificate expiry to the migration plan",
         "owner": "Unassigned", "due": "Before the 1st", "priority": "medium",
         "context": "It is currently only mentioned in conversation."},
    ],
    "priorities": [
        {"point": "Data loss in the dual-write window",
         "priority": "high",
         "reason": "It is the only item that can lose customer data, and the "
                   "fix is not written yet."},
        {"point": "The certificate expiry on the 1st",
         "priority": "high",
         "reason": "A hard external deadline that constrains every other date."},
        {"point": "Choosing the cutover date",
         "priority": "medium",
         "reason": "Blocked on a conversation with Marcus rather than on work."},
        {"point": "Staffing during the cutover week",
         "priority": "low",
         "reason": "Resolves itself if the date moves to the 19th."},
    ],
    "speakers": [
        {"speaker": "Priya Raman",
         "summary": "Chaired the meeting and drove both decisions to a close.",
         "key_points": ["Pressed for a realistic size for the dual-write window",
                        "Took on the conversation with Marcus about the date",
                        "Asked for the certificate deadline to reach the plan"]},
        {"speaker": "Daniel Okafor",
         "summary": "Owns the migration work and raised the two technical risks.",
         "key_points": ["Read path stable in staging for four days",
                        "Dual-write window can reach six minutes at peak",
                        "Flagged the certificate expiry as a hard deadline"]},
        {"speaker": "Sofia Lindqvist",
         "summary": "Brought the precedent that shaped the agreed solution.",
         "key_points": ["Hit the same problem during the billing migration",
                        "Offered the existing reconciliation code",
                        "Raised the staffing problem with the 12th"]},
    ],
    "drafts": [
        {"kind": "email", "audience": "Priya, Daniel, Sofia",
         "subject": "Atlas cutover: what we agreed",
         "body": "Hi all,\n\nThanks for the time this morning. Quick recap so "
                 "nothing gets lost.\n\nWe agreed the dual-write window is the "
                 "main risk, and that a reconciliation job replaying dropped "
                 "rows is the way to handle it — the same approach that worked "
                 "for billing last year.\n\nWho owes what:\n"
                 "- Daniel: reconciliation job, by Wednesday\n"
                 "- Sofia: send Daniel the billing migration code this afternoon\n"
                 "- Priya: make the case to Marcus for cutting over on the 19th\n\n"
                 "One open item: the old cluster's certificate expires on the "
                 "1st. That's a hard deadline whichever date we land on, so I'll "
                 "add it to the plan.\n\nShout if I've misremembered anything.\n\n"
                 "[Your name]"},
        {"kind": "chat", "audience": "#atlas-migration",
         "subject": "",
         "body": "Cutover sync recap: reconciliation job is the agreed fix for "
                 "the dual-write window (Daniel, by Wed). Sofia's sending over "
                 "the billing migration code this afternoon. Priya's pushing "
                 "Marcus for the 19th rather than the 12th — we're short-staffed "
                 "that week. Heads up that the old cluster cert expires on the "
                 "1st, so that's the real deadline."},
    ],
}


class _StubOllama(BaseHTTPRequestHandler):
    """Answers /api/tags so the Setup panel shows a configured machine.

    The screenshots are meant to show what the dashboard looks like on a
    machine that has been set up, and this sandbox has no Ollama. The health
    check itself is the real one -- it is just pointed at this stub.
    """

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
        body = json.dumps(
            {"models": [{"name": "qwen2.5:7b-instruct"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_stub_ollama() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), _StubOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def optimise(path: Path) -> None:
    """Shrink a screenshot so the repository does not carry megabytes of PNG."""
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(path) as image:
        image.convert("RGB").quantize(colors=256, method=Image.MEDIANCUT).save(
            path, optimize=True)


def seed(service) -> int:
    """Create the demo meetings and return the id of the interesting one."""
    repo = service.repo
    service.acknowledge_consent(True)

    meeting_id = repo.create_meeting("Atlas migration: cutover planning",
                                     source="auto", participants_informed=True,
                                     started_at="2026-09-15T09:30:00Z")
    audio = write_wav(
        service.settings.audio_dir / f"meeting-{meeting_id:06d}" / "mixed.wav",
        __import__("numpy").zeros(16000, dtype="float32"), 16000)
    repo.update_meeting(meeting_id, status=STATUS_COMPLETE, duration_seconds=1147,
                        ended_at="2026-09-15T09:49:07Z", audio_path=str(audio))

    labels = {"Priya Raman": "Speaker 1", "Daniel Okafor": "Speaker 2",
              "Sofia Lindqvist": "Speaker 3"}
    repo.replace_transcript(meeting_id, [
        TranscriptSegment(start, end, text, speaker=labels[name])
        for start, end, name, text in TRANSCRIPT
    ])

    totals = {name: 0.0 for name in labels}
    turns = {name: 0 for name in labels}
    words = {name: 0 for name in labels}
    for start, end, name, text in TRANSCRIPT:
        totals[name] += end - start
        turns[name] += 1
        words[name] += len(text.split())
    grand = sum(totals.values())

    by_name = {entry["speaker"]: entry for entry in NOTES["speakers"]}
    repo.replace_speakers(meeting_id, [
        {"label": labels[name], "display_name": name,
         "talk_seconds": round(totals[name] * 14.5, 1),
         "word_count": words[name] * 12, "turn_count": turns[name] * 3,
         "share": totals[name] / grand,
         "summary": by_name[name]["summary"],
         "key_points": by_name[name]["key_points"]}
        for name in ("Priya Raman", "Daniel Okafor", "Sofia Lindqvist")
    ])
    repo.save_intelligence(meeting_id, MeetingIntelligence.model_validate(NOTES),
                           "qwen2.5:7b-instruct")

    # A couple of neighbours so the list page is not a single row.
    second = repo.create_meeting("Design review: notification preferences",
                                 participants_informed=True,
                                 started_at="2026-09-14T14:00:00Z")
    repo.update_meeting(second, status=STATUS_COMPLETE, duration_seconds=1892,
                        ended_at="2026-09-14T14:31:32Z")
    repo.save_intelligence(second, MeetingIntelligence.model_validate({
        "summary": "Walked through the notification settings redesign. Agreed "
                   "to drop the per-channel toggles in favour of three presets, "
                   "with an advanced view for the people who need it.",
    }), "qwen2.5:7b-instruct")

    third = repo.create_meeting("Weekly standup", source="auto",
                                started_at="2026-09-14T09:15:00Z")
    repo.update_meeting(third, status=STATUS_PROCESSING, duration_seconds=624,
                        stage="Identifying speakers", progress=0.62)

    return meeting_id


def _chromium_executable() -> str | None:
    """Use a preinstalled Chromium if Playwright's own build is not there.

    Sandboxes and CI images often ship one Chromium at a fixed path rather than
    the exact build the installed Playwright expects.
    """
    import os

    candidates = [os.environ.get("CHROMIUM_PATH"), "/opt/pw-browsers/chromium"]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def capture(runtime: AppRuntime, meeting_id: int) -> None:
    from playwright.sync_api import sync_playwright

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = runtime.dashboard_url.rstrip("/")
    executable = _chromium_executable()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=executable)
        for scheme, suffix in (("light", ""), ("dark", "-dark")):
            context = browser.new_context(viewport=VIEWPORT,
                                          device_scale_factor=2,
                                          color_scheme=scheme)
            page = context.new_page()
            page.goto(runtime.authorised_url(), wait_until="networkidle")
            page.wait_for_selector(".meeting-list li", timeout=15_000)
            time.sleep(0.6)
            page.screenshot(path=str(OUTPUT_DIR / f"dashboard-meetings{suffix}.png"),
                            full_page=True)

            page.goto(f"{base}/meeting/{meeting_id}", wait_until="networkidle")
            page.wait_for_selector("#transcript-list .line", timeout=15_000)
            time.sleep(0.6)
            page.screenshot(path=str(OUTPUT_DIR / f"dashboard-meeting{suffix}.png"),
                            full_page=True)
            context.close()
        browser.close()


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="teams-notes-shots-"))
    data_dir = workspace / "teams-notes" / "data"
    stub, stub_url = start_stub_ollama()
    os.environ["OLLAMA_HOST"] = stub_url
    os.environ["HF_TOKEN"] = "hf_screenshot_placeholder_not_a_real_token"
    try:
        settings = load_settings(env_file=Path("/nonexistent"), data_dir=data_dir)
        runtime = AppRuntime(settings)
        runtime.start(with_monitor=False)
        try:
            meeting_id = seed(runtime.service)
            capture(runtime, meeting_id)
        finally:
            runtime.stop()
    finally:
        stub.shutdown()
        shutil.rmtree(workspace, ignore_errors=True)

    for path in sorted(OUTPUT_DIR.glob("dashboard-*.png")):
        optimise(path)
        print(f"  {path.relative_to(PROJECT_ROOT)}  "
              f"{path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
