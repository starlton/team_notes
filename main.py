#!/usr/bin/env python3
"""Teams Notes command line.

    python main.py tray          Run in the background with a tray icon (normal use)
    python main.py dashboard     Serve the dashboard only, and open it
    python main.py record        Record from the CLI until you press Enter
    python main.py import a.wav  Bring an existing recording in and process it
    python main.py process 3     Re-run the processing pass for meeting 3
    python main.py meetings      List meetings
    python main.py devices       List the audio devices that were found
    python main.py doctor        Check that Ollama, the token and the models are set up

Run `python main.py <command> --help` for the options on each.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from app.errors import TeamsNotesError
from app.logging_setup import get_logger, setup_logging
from app.paths import meeting_audio_dir
from app.pipeline.service import CONSENT_NOTICE, MeetingService
from app.transcribe.models import format_timestamp
from config import ConfigError, Settings, get_settings

log = get_logger(__name__)


# --- helpers ----------------------------------------------------------------

def _make_service(settings: Settings) -> MeetingService:
    service = MeetingService(settings)
    service.start()
    return service


def _ensure_consent(service: MeetingService, auto_accept: bool) -> bool:
    """Show the recording notice once and record that it was acknowledged."""
    if service.consent_acknowledged:
        return True

    print("\n" + "-" * 72)
    print(CONSENT_NOTICE)
    print("-" * 72)
    if auto_accept:
        service.acknowledge_consent(True)
        print("Acknowledged (--accept-notice).\n")
        return True

    try:
        answer = input("Type 'yes' to confirm you understand: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if answer in {"yes", "y"}:
        service.acknowledge_consent(True)
        print()
        return True
    print("Not acknowledged, so nothing will be recorded.")
    return False


def _print_meeting_row(meeting: dict) -> None:
    flags = []
    if meeting.get("source") == "auto":
        flags.append("auto")
    if meeting.get("participants_informed"):
        flags.append("informed")
    suffix = f" [{', '.join(flags)}]" if flags else ""
    print(f"  {meeting['id']:>4}  {str(meeting.get('started_at'))[:16]}  "
          f"{format_timestamp(meeting.get('duration_seconds') or 0):>7}  "
          f"{str(meeting.get('status')):<11} {meeting.get('title') or '(untitled)'}"
          f"{suffix}")


# --- commands ---------------------------------------------------------------

def cmd_tray(args: argparse.Namespace, settings: Settings) -> int:
    from app.tray.trayapp import TrayApp

    app = TrayApp()
    if not _ensure_consent(app.service, args.accept_notice):
        return 1
    print(f"Teams Notes is running. Dashboard: {settings.dashboard_url}")
    print("Use the tray icon to start and stop recording, or open the dashboard.")
    app.run()
    return 0


def cmd_dashboard(args: argparse.Namespace, settings: Settings) -> int:
    from app.runtime import AppRuntime

    runtime = AppRuntime(settings)
    runtime.start(with_monitor=not args.no_detect)
    url = runtime.authorised_url()
    print(f"Dashboard: {url}")
    print("That link carries this session's access token. Keep it to yourself.")
    if not args.no_browser:
        runtime.open_dashboard()
    print("Press Ctrl+C to stop.")
    try:
        while True:
            import time

            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        runtime.stop()
    return 0


def cmd_record(args: argparse.Namespace, settings: Settings) -> int:
    service = _make_service(settings)
    if not _ensure_consent(service, args.accept_notice):
        return 1

    try:
        meeting_id = service.start_recording(
            title=args.title, source="cli",
            participants_informed=args.informed)
    except TeamsNotesError as exc:
        print(f"Could not start recording: {exc}", file=sys.stderr)
        return 1

    print(f"Recording meeting {meeting_id}. Both the meeting audio and your "
          f"microphone are being captured.")
    try:
        if args.seconds:
            import time

            print(f"Stopping automatically after {args.seconds}s. Ctrl+C to stop now.")
            time.sleep(args.seconds)
        else:
            input("Press Enter to stop… ")
    except KeyboardInterrupt:
        print()

    try:
        result = service.stop_recording(process=not args.no_process)
    except TeamsNotesError as exc:
        print(f"Could not stop cleanly: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {result['duration_seconds']:.1f}s to {result['audio_path']}")
    for warning in result.get("warnings", []):
        print(f"  note: {warning}")

    if args.no_process:
        print(f"Process it later with: python main.py process {meeting_id}")
        return 0
    return _wait_for_processing(service, meeting_id)


def cmd_import(args: argparse.Namespace, settings: Settings) -> int:
    """Bring an existing WAV in as a meeting and run the processing pass."""
    source = Path(args.wav).expanduser().resolve()
    if not source.is_file():
        print(f"No such file: {source}", file=sys.stderr)
        return 1

    from app.capture.wav_io import wav_duration_seconds

    service = _make_service(settings)
    duration = wav_duration_seconds(source)
    if duration <= 0:
        print(f"{source.name} is not a readable WAV file.", file=sys.stderr)
        return 1

    meeting_id = service.repo.create_meeting(
        title=args.title or source.stem, source="import",
        participants_informed=args.informed)
    destination = meeting_audio_dir(settings.audio_dir, meeting_id) / "mixed.wav"
    shutil.copy2(source, destination)
    service.repo.update_meeting(
        meeting_id, status="recorded", audio_path=str(destination),
        duration_seconds=duration, ended_at=None)

    print(f"Imported {source.name} ({duration:.1f}s) as meeting {meeting_id}.")
    if args.no_process:
        return 0
    service.queue_processing(meeting_id)
    return _wait_for_processing(service, meeting_id)


def cmd_process(args: argparse.Namespace, settings: Settings) -> int:
    service = _make_service(settings)
    meeting_id = args.meeting_id
    if meeting_id is None:
        meetings = service.repo.list_meetings(limit=1)
        if not meetings:
            print("There are no meetings to process.", file=sys.stderr)
            return 1
        meeting_id = int(meetings[0]["id"])
        print(f"Processing the most recent meeting ({meeting_id}).")

    try:
        service.queue_processing(meeting_id)
    except TeamsNotesError as exc:
        print(f"Could not queue meeting {meeting_id}: {exc}", file=sys.stderr)
        return 1
    return _wait_for_processing(service, meeting_id)


def _wait_for_processing(service: MeetingService, meeting_id: int) -> int:
    """Block until the queued job finishes, printing progress as it goes."""
    import time

    print("Processing. This runs locally and can take a minute or two.")
    last_stage = ""
    # If the worker dies without updating the meeting, the status would sit at
    # "recorded" forever. Give up after a few idle polls rather than hanging.
    idle_polls = 0
    while True:
        time.sleep(1.0)
        meeting = service.repo.find_meeting(meeting_id)
        if meeting is None:
            print("The meeting disappeared while processing.", file=sys.stderr)
            return 1

        stage = str(meeting.get("stage") or "")
        if stage and stage != last_stage:
            print(f"  {int((meeting.get('progress') or 0) * 100):>3}%  {stage}")
            last_stage = stage

        status = meeting.get("status")
        if status == "complete":
            break
        if status == "failed":
            print(f"\nProcessing failed: {meeting.get('error')}", file=sys.stderr)
            return 1

        if service.jobs.is_busy():
            idle_polls = 0
            continue

        idle_polls += 1
        if idle_polls >= 5:
            state = service.jobs.job_state(f"process-{meeting_id}")
            print(f"\nProcessing stopped without finishing (job state: {state}).",
                  file=sys.stderr)
            print("Check data/logs/teams-notes.log for the reason.", file=sys.stderr)
            return 1

    summary = service.repo.get_summary(meeting_id) or {}
    actions = service.repo.get_action_items(meeting_id)
    speakers = service.repo.get_speakers(meeting_id)

    print(f"\n=== {summary.get('title') or meeting.get('title')} ===")
    if summary.get("summary"):
        print(f"\n{summary['summary']}\n")
    for bullet in summary.get("bullets", []):
        print(f"  • {bullet}")
    if actions:
        print("\nTo-dos:")
        for item in actions:
            due = f", due {item['due']}" if item["due"] else ""
            print(f"  [{item['priority']}] {item['task']} — {item['owner']}{due}")
    if speakers:
        print("\nSpeakers:")
        for speaker in speakers:
            print(f"  {speaker['display_name']}: "
                  f"{format_timestamp(speaker['talk_seconds'])} "
                  f"({round(speaker['share'] * 100)}%)")
    for warning in meeting.get("warnings", []):
        print(f"\n  note: {warning}")

    print(f"\nFull notes: {service.settings.dashboard_url}meeting/{meeting_id}")
    return 0


def cmd_meetings(args: argparse.Namespace, settings: Settings) -> int:
    service = _make_service(settings)
    meetings = service.repo.list_meetings(limit=args.limit)
    if not meetings:
        print("No meetings yet.")
        return 0
    print(f"  {'id':>4}  {'started':<16}  {'length':>7}  status      title")
    for meeting in meetings:
        _print_meeting_row(meeting)
    return 0


def cmd_devices(_args: argparse.Namespace, _settings: Settings) -> int:
    from app.capture.devices import AudioHost

    try:
        with AudioHost() as host:
            devices = host.list_devices()
            if not devices:
                print("No input devices were found.")
                return 1
            print("Input devices:")
            for device in devices:
                print(f"  {device.describe()}")
            print(f"\nLoopback that will be used: {host.find_loopback().describe()}")
            print(f"Microphone that will be used: {host.find_microphone().describe()}")
    except TeamsNotesError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    return 0


def cmd_doctor(_args: argparse.Namespace, settings: Settings) -> int:
    service = _make_service(settings)
    health = service.health()
    ok = True

    print("Teams Notes setup check\n")
    print(f"  Data directory:  {settings.data_dir}")
    print(f"  Whisper model:   {settings.whisper_model} ({settings.whisper_compute_type})")

    ollama = health["ollama"]
    if ollama["ok"]:
        print(f"  Ollama:          ready ({ollama['model']})")
    else:
        ok = False
        print(f"  Ollama:          NOT READY — {ollama['error']}")
        print(f"                   {ollama['remedy']}")

    diarization = health["diarization"]
    if diarization["ok"]:
        print("  Speaker labels:  ready (Hugging Face token found)")
    else:
        print(f"  Speaker labels:  off — {diarization['error']}")
        print(f"                   {diarization['remedy']}")

    try:
        from app.capture.devices import AudioHost

        with AudioHost() as host:
            print(f"  Loopback audio:  {host.find_loopback().name}")
            print(f"  Microphone:      {host.find_microphone().name}")
    except TeamsNotesError as exc:
        print(f"  Audio capture:   NOT READY — {exc.message}")
        print(f"                   {exc.remedy}")
        ok = False

    consent = "acknowledged" if service.consent_acknowledged else "NOT acknowledged yet"
    print(f"  Recording notice: {consent}")
    print(f"\n  Meetings stored: {health['storage']['meetings']}")
    print("\nAll good." if ok else "\nSome things need attention (see above).")
    return 0 if ok else 1


def cmd_consent(args: argparse.Namespace, settings: Settings) -> int:
    service = _make_service(settings)
    if args.reset:
        service.acknowledge_consent(False)
        print("The recording notice will be shown again before the next recording.")
        return 0
    _ensure_consent(service, args.accept_notice)
    return 0


# --- argument parsing --------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="teams-notes",
        description="A fully local meeting assistant for Microsoft Teams.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    tray = sub.add_parser("tray", help="run in the background with a tray icon")
    tray.add_argument("--accept-notice", action="store_true",
                      help="acknowledge the recording notice without prompting")
    tray.set_defaults(func=cmd_tray)

    dashboard = sub.add_parser("dashboard", help="serve the dashboard only")
    dashboard.add_argument("--no-browser", action="store_true",
                           help="do not open a browser window")
    dashboard.add_argument("--no-detect", action="store_true",
                           help="do not watch for Teams calls")
    dashboard.set_defaults(func=cmd_dashboard)

    record = sub.add_parser("record", help="record until you press Enter")
    record.add_argument("--title", default="", help="a title for the meeting")
    record.add_argument("--seconds", type=int, default=0,
                        help="stop automatically after this many seconds")
    record.add_argument("--no-process", action="store_true",
                        help="save the audio without processing it")
    record.add_argument("--informed", action="store_true",
                        help="record that participants were told")
    record.add_argument("--accept-notice", action="store_true",
                        help="acknowledge the recording notice without prompting")
    record.set_defaults(func=cmd_record)

    import_cmd = sub.add_parser("import", help="process an existing WAV file")
    import_cmd.add_argument("wav", help="path to a WAV file")
    import_cmd.add_argument("--title", default="", help="a title for the meeting")
    import_cmd.add_argument("--informed", action="store_true",
                            help="record that participants were told")
    import_cmd.add_argument("--no-process", action="store_true",
                            help="import without processing")
    import_cmd.set_defaults(func=cmd_import)

    process = sub.add_parser("process", help="run the processing pass for a meeting")
    process.add_argument("meeting_id", nargs="?", type=int, default=None,
                         help="meeting id (default: the most recent)")
    process.set_defaults(func=cmd_process)

    meetings = sub.add_parser("meetings", help="list recorded meetings")
    meetings.add_argument("--limit", type=int, default=25)
    meetings.set_defaults(func=cmd_meetings)

    devices = sub.add_parser("devices", help="list audio devices")
    devices.set_defaults(func=cmd_devices)

    doctor = sub.add_parser("doctor", help="check the setup")
    doctor.set_defaults(func=cmd_doctor)

    consent = sub.add_parser("consent", help="show or reset the recording notice")
    consent.add_argument("--reset", action="store_true",
                         help="require the notice to be acknowledged again")
    consent.add_argument("--accept-notice", action="store_true")
    consent.set_defaults(func=cmd_consent)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        # No subcommand: running the app in the background is the normal use.
        args = parser.parse_args(["tray"])

    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"Configuration problem: {exc}", file=sys.stderr)
        print("Check your .env file against .env.example.", file=sys.stderr)
        return 2

    settings.ensure_dirs()
    setup_logging(settings.log_dir, settings.log_level)

    try:
        return int(args.func(args, settings))
    except TeamsNotesError as exc:
        print(f"\n{exc.message}", file=sys.stderr)
        if exc.remedy:
            print(f"{exc.remedy}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
