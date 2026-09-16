"""The system-tray app.

The tray icon is deliberately explicit about what the app is doing: the icon
turns red and the tooltip says "Recording" whenever audio is being captured.
Consent is a feature here too -- the tool should never be mistakable for
something running secretly.
"""

from __future__ import annotations

import threading
from typing import Any

from app.errors import DependencyMissingError, TeamsNotesError
from app.logging_setup import get_logger
from app.runtime import AppRuntime

log = get_logger(__name__)

ICON_SIZE = 64


def _import_pystray() -> tuple[Any, Any]:
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise DependencyMissingError(
            "The tray app needs 'pystray' and 'pillow'.",
            "Activate the venv and run: pip install pystray pillow",
        ) from exc
    return pystray, (Image, ImageDraw)


def _make_icon(recording: bool) -> Any:
    """Draw the tray icon: a blue dot normally, a red one while recording."""
    _, (Image, ImageDraw) = _import_pystray()
    image = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    fill = (220, 60, 50, 255) if recording else (47, 111, 235, 255)
    draw.ellipse((6, 6, ICON_SIZE - 6, ICON_SIZE - 6), fill=fill)
    if recording:
        draw.ellipse((24, 24, ICON_SIZE - 24, ICON_SIZE - 24), fill=(255, 255, 255, 255))
    else:
        draw.rectangle((28, 20, 36, 40), fill=(255, 255, 255, 255))
        draw.ellipse((22, 34, 42, 48), outline=(255, 255, 255, 255), width=4)
    return image


class TrayApp:
    """Runs the app in the background with a tray icon driving it."""

    def __init__(self, runtime: AppRuntime | None = None) -> None:
        self.runtime = runtime or AppRuntime()
        self.service = self.runtime.service
        self._icon: Any = None
        self._refresh_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_error = ""

    # --- menu actions -----------------------------------------------------

    def _toggle_recording(self) -> None:
        try:
            if self.service.is_recording:
                result = self.service.stop_recording(process=True)
                self._notify("Recording stopped",
                             f"Processing meeting {result['meeting_id']}…")
            else:
                meeting_id = self.service.start_recording(source="tray")
                self._notify("Recording started",
                             f"Meeting {meeting_id}. Participants should be told.")
            self._last_error = ""
        except TeamsNotesError as exc:
            self._last_error = str(exc)
            self._notify("Teams Notes", str(exc))
            log.warning("Tray action failed: %s", exc)

    def _discard(self) -> None:
        try:
            meeting_id = self.service.discard_recording()
            if meeting_id is not None:
                self._notify("Recording discarded", "The audio was deleted.")
        except TeamsNotesError as exc:
            self._notify("Teams Notes", str(exc))

    def _open_dashboard(self) -> None:
        self.runtime.open_dashboard()

    def _toggle_auto_detect(self) -> None:
        enabled = not self.service.auto_detect_enabled
        self.service.set_auto_detect(enabled)
        if enabled:
            self.runtime.monitor.start()
        else:
            self.runtime.monitor.stop()

    def _quit(self) -> None:
        self._stop.set()
        if self._icon is not None:
            self._icon.visible = False
            self._icon.stop()

    # --- menu -------------------------------------------------------------

    def _build_menu(self) -> Any:
        pystray, _ = _import_pystray()
        item = pystray.MenuItem

        def recording_label(_item: Any) -> str:
            return "Stop recording & process" if self.service.is_recording \
                else "Start recording"

        def status_label(_item: Any) -> str:
            status = self.service.status()
            if self._last_error:
                return f"⚠ {self._last_error[:60]}"
            if status["recording"]:
                minutes, seconds = divmod(int(status["elapsed_seconds"]), 60)
                return f"● Recording — {minutes}:{seconds:02d}"
            jobs = status["jobs"]
            if jobs.get("running"):
                return f"⚙ {jobs['running']['description']}"
            return "Idle"

        return pystray.Menu(
            item(status_label, None, enabled=False),
            pystray.Menu.SEPARATOR,
            item(recording_label, lambda: self._toggle_recording()),
            item("Discard current recording", lambda: self._discard(),
                 enabled=lambda _i: self.service.is_recording),
            pystray.Menu.SEPARATOR,
            item("Open dashboard", lambda: self._open_dashboard(), default=True),
            item("Detect Teams calls automatically",
                 lambda: self._toggle_auto_detect(),
                 checked=lambda _i: self.service.auto_detect_enabled),
            pystray.Menu.SEPARATOR,
            item("Quit", lambda: self._quit()),
        )

    def _notify(self, title: str, message: str) -> None:
        if self._icon is None:
            log.info("%s: %s", title, message)
            return
        try:
            self._icon.notify(message, title)
        except Exception:  # pragma: no cover - notifications are optional
            log.info("%s: %s", title, message)

    # --- refresh loop -----------------------------------------------------

    def _refresh_loop(self) -> None:
        recording = None
        while not self._stop.wait(1.0):
            try:
                now_recording = self.service.is_recording
                if now_recording != recording:
                    recording = now_recording
                    if self._icon is not None:
                        self._icon.icon = _make_icon(now_recording)
                        self._icon.title = ("Teams Notes — RECORDING"
                                            if now_recording else "Teams Notes")
                if self._icon is not None:
                    self._icon.update_menu()
            except Exception:  # noqa: BLE001 - the tray must never die
                log.debug("Tray refresh failed", exc_info=True)

    # --- run --------------------------------------------------------------

    def run(self) -> None:
        """Start everything and block on the tray icon's event loop."""
        pystray, _ = _import_pystray()
        self.runtime.start()

        self._icon = pystray.Icon(
            "teams-notes", _make_icon(False), "Teams Notes", self._build_menu())

        self._refresh_thread = threading.Thread(target=self._refresh_loop,
                                                name="tray-refresh", daemon=True)
        self._refresh_thread.start()

        if self.runtime.settings.open_browser_on_start:
            self.runtime.open_dashboard()

        log.info("Tray app running. Dashboard: %s", self.runtime.dashboard_url)
        try:
            self._icon.run()
        finally:
            self._stop.set()
            self.runtime.stop()
