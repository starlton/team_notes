"""Starting and stopping the whole app.

One object owns the service, the web server thread and the Teams monitor, so
the tray, the CLI and the tests all bring the app up the same way and take it
down in the same order.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from app.autodetect.teams_monitor import TeamsMonitor
from app.logging_setup import get_logger, setup_logging
from app.pipeline.service import MeetingService
from app.web.security import TokenStore
from app.web.server import create_app
from config import Settings, get_settings

log = get_logger(__name__)

SERVER_START_TIMEOUT_SECONDS = 15.0


class AppRuntime:
    """Owns the service, the HTTP server and the meeting detector."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        setup_logging(self.settings.log_dir, self.settings.log_level)

        self.service = MeetingService(self.settings)
        self.app = create_app(self.service)
        self.tokens = TokenStore(self.settings.runtime_dir)
        self.monitor = TeamsMonitor(self.service)

        self._server: Any = None
        self._server_thread: threading.Thread | None = None

    # --- urls -------------------------------------------------------------

    @property
    def dashboard_url(self) -> str:
        return self.settings.dashboard_url

    def authorised_url(self) -> str:
        """A dashboard link that carries the token, for opening a browser.

        /auth is the only endpoint that accepts a token in the query string; it
        swaps it for a cookie and redirects, so the token does not stay in the
        address bar.
        """
        token = self.tokens.load_or_create()
        return f"{self.settings.dashboard_url.rstrip('/')}/auth?token={token}"

    # --- lifecycle --------------------------------------------------------

    def start(self, with_monitor: bool = True) -> None:
        """Start background services and the local web server."""
        self.service.start()
        self._start_server()
        if with_monitor and self.service.auto_detect_enabled:
            self.monitor.start()

    def stop(self) -> None:
        """Stop everything, finishing an in-flight recording first."""
        log.info("Shutting down")
        try:
            self.monitor.stop()
        except Exception:  # pragma: no cover - best effort
            log.debug("Monitor shutdown raised", exc_info=True)
        try:
            self.service.shutdown()
        finally:
            self._stop_server()

    # --- web server -------------------------------------------------------

    def _start_server(self) -> None:
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.settings.web_host,
            port=self.settings.web_port,
            log_level=self.settings.log_level.lower(),
            # The dashboard token travels in the /auth query string, so access
            # logging is off to keep it out of the log file.
            access_log=False,
            timeout_graceful_shutdown=5,
        )
        self._server = uvicorn.Server(config)
        # uvicorn installs signal handlers by default, which only works on the
        # main thread; this server runs on a worker thread.
        self._server.install_signal_handlers = lambda: None

        self._server_thread = threading.Thread(target=self._server.run,
                                               name="web-server", daemon=True)
        self._server_thread.start()

        deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if getattr(self._server, "started", False):
                log.info("Dashboard is live at %s", self.dashboard_url)
                return
            if not self._server_thread.is_alive():
                raise RuntimeError(
                    f"The dashboard server stopped immediately. Port "
                    f"{self.settings.web_port} may already be in use; set "
                    f"WEB_PORT in your .env file to a free port."
                )
            time.sleep(0.05)
        raise RuntimeError(
            f"The dashboard server did not start within "
            f"{SERVER_START_TIMEOUT_SECONDS:.0f}s."
        )

    def _stop_server(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        thread = self._server_thread
        if thread is not None:
            thread.join(timeout=10.0)
        self._server_thread = None
        self._server = None

    def open_dashboard(self) -> None:
        """Open the dashboard in the default browser, already authorised."""
        import webbrowser

        url = self.authorised_url()
        log.info("Opening the dashboard")
        webbrowser.open(url)
