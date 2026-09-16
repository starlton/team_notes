"""Security for the local dashboard.

"It only listens on localhost" is not by itself a security boundary. On a shared
or work machine, every other program running as any user can reach 127.0.0.1,
and any web page the user visits can make requests to it. This dashboard holds
recordings of private conversations, so it defends against both:

* **A shared secret.** A random token is minted at startup and stored in a file
  only the user can read. Requests must present it, so another local process
  cannot simply GET the transcripts.
* **Host header checks.** Without them, a site the user visits can point a
  hostname it controls at 127.0.0.1 (DNS rebinding) and read responses from this
  server despite the same-origin policy. Only loopback Host values are accepted.
* **SameSite=Strict cookies plus a required custom header.** A cross-site form
  post cannot set a custom header without a CORS preflight the server refuses,
  so it cannot drive state-changing endpoints even if a cookie leaked.
* **A strict CSP with no inline script.** The dashboard renders text produced by
  a language model from whatever people said in a meeting. Everything is written
  via textContent rather than innerHTML, and the CSP is the backstop.
"""

from __future__ import annotations

import hmac
import os
import secrets
import threading
import time
from pathlib import Path

from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from app.logging_setup import get_logger

log = get_logger(__name__)

COOKIE_NAME = "teams_notes_session"
TOKEN_HEADER = "x-auth-token"
CSRF_HEADER = "x-requested-with"
CSRF_VALUE = "teams-notes"
TOKEN_FILENAME = "dashboard-token"

# Paths reachable without a token. Deliberately tiny.
PUBLIC_PATHS = frozenset({"/auth", "/favicon.ico"})

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "media-src 'self'; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)

MAX_FAILED_ATTEMPTS = 20
FAILURE_WINDOW_SECONDS = 60.0


class TokenStore:
    """Mints and persists the dashboard token."""

    def __init__(self, runtime_dir: Path) -> None:
        self.path = Path(runtime_dir) / TOKEN_FILENAME
        self._token = ""

    def load_or_create(self) -> str:
        """Reuse an existing token so open dashboard tabs survive a restart."""
        if self._token:
            return self._token

        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._read()
        if existing:
            self._token = existing
            return existing

        token = secrets.token_urlsafe(32)
        self._write(token)
        self._token = token
        log.info("Created a new dashboard token at %s", self.path)
        return token

    def rotate(self) -> str:
        token = secrets.token_urlsafe(32)
        self._write(token)
        self._token = token
        return token

    def _read(self) -> str:
        try:
            token = self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        # A short token means a truncated or tampered file; mint a fresh one.
        return token if len(token) >= 32 else ""

    def _write(self, token: str) -> None:
        temp = self.path.with_suffix(".tmp")
        temp.write_text(token, encoding="utf-8")
        try:
            if os.name == "posix":
                os.chmod(temp, 0o600)
        except OSError:  # pragma: no cover - platform dependent
            pass
        os.replace(temp, self.path)


class FailureLimiter:
    """Slows down repeated bad tokens. The token is long, but be tidy anyway."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failures: list[float] = []

    def record_failure(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._failures = [t for t in self._failures
                              if now - t < FAILURE_WINDOW_SECONDS]
            self._failures.append(now)

    def is_blocked(self) -> bool:
        with self._lock:
            now = time.monotonic()
            self._failures = [t for t in self._failures
                              if now - t < FAILURE_WINDOW_SECONDS]
            return len(self._failures) >= MAX_FAILED_ATTEMPTS

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()


def allowed_hosts(port: int) -> frozenset[str]:
    """Host header values this server will answer to."""
    names = ("127.0.0.1", "localhost", "[::1]")
    values = {f"{name}:{port}" for name in names}
    values.update(names)  # a Host without an explicit port
    return frozenset(values)


def host_is_allowed(headers: Headers, port: int) -> bool:
    host = (headers.get("host") or "").strip().lower()
    if not host:
        return False
    return host in allowed_hosts(port)


def token_matches(candidate: str, token: str) -> bool:
    """Constant-time comparison, so timing cannot reveal the token."""
    if not candidate or not token:
        return False
    return hmac.compare_digest(candidate.encode("utf-8"), token.encode("utf-8"))


def apply_security_headers(response: Response) -> Response:
    response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=(), interest-cohort=()"
    )
    # Nothing here should ever be cached by an intermediary; it is all private.
    response.headers.setdefault("Cache-Control", "no-store")
    return response


class SecurityMiddleware:
    """Host validation, token auth, CSRF checks and security headers."""

    def __init__(self, app: ASGIApp, token_store: TokenStore, port: int,
                 limiter: FailureLimiter | None = None) -> None:
        self.app = app
        self.tokens = token_store
        self.port = port
        # Shared with the /auth route so a bad link and a bad header count
        # towards the same throttle.
        self.limiter = limiter or FailureLimiter()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        denial = self._check(request)
        if denial is not None:
            await apply_security_headers(denial)(scope, receive, send)
            return

        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                existing = {name.lower() for name, _ in headers}
                for name, value in _SECURITY_HEADER_PAIRS:
                    if name not in existing:
                        headers.append((name, value))
            await send(message)

        await self.app(scope, receive, send_with_headers)

    # --- checks -----------------------------------------------------------

    def _check(self, request: Request) -> Response | None:
        if not host_is_allowed(request.headers, self.port):
            log.warning("Rejected a request with Host=%r",
                        request.headers.get("host"))
            return JSONResponse(
                {"error": "This dashboard only answers requests addressed to "
                          "localhost."},
                status_code=421,
            )

        path = request.url.path
        if path in PUBLIC_PATHS:
            # /auth takes a token in the query string, so it gets the same
            # throttle as the header path even though it is otherwise public.
            if self.limiter.is_blocked():
                return JSONResponse(
                    {"error": "Too many failed attempts. Wait a minute and retry."},
                    status_code=429,
                )
            return None

        token = self.tokens.load_or_create()
        if not self._authenticated(request, token):
            if self.limiter.is_blocked():
                return JSONResponse(
                    {"error": "Too many failed attempts. Wait a minute and retry."},
                    status_code=429,
                )
            self.limiter.record_failure()
            return JSONResponse(
                {"error": "Not authorised.",
                 "remedy": "Open the dashboard from the tray menu, or run "
                           "'python main.py dashboard' to get an authorised link."},
                status_code=401,
            )
        self.limiter.reset()

        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            csrf_error = self._check_csrf(request)
            if csrf_error is not None:
                return csrf_error
        return None

    def _authenticated(self, request: Request, token: str) -> bool:
        header_token = request.headers.get(TOKEN_HEADER, "")
        if token_matches(header_token, token):
            return True
        return token_matches(request.cookies.get(COOKIE_NAME, ""), token)

    def _check_csrf(self, request: Request) -> Response | None:
        """Reject writes that a cross-site page could have triggered."""
        if request.headers.get(CSRF_HEADER, "").lower() != CSRF_VALUE:
            return JSONResponse(
                {"error": "Missing the request header this dashboard requires for "
                          "changes."},
                status_code=403,
            )

        origin = request.headers.get("origin")
        if origin:
            allowed = {f"http://{name}" for name in
                       ("127.0.0.1", "localhost", "[::1]")}
            allowed |= {f"http://{name}:{self.port}" for name in
                        ("127.0.0.1", "localhost", "[::1]")}
            if origin not in allowed:
                log.warning("Rejected a cross-origin write from %r", origin)
                return JSONResponse({"error": "Cross-origin requests are refused."},
                                    status_code=403)
        return None


_SECURITY_HEADER_PAIRS = [
    (b"content-security-policy", CONTENT_SECURITY_POLICY.encode()),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"permissions-policy",
     b"camera=(), microphone=(), geolocation=(), interest-cohort=()"),
    (b"cache-control", b"no-store"),
]
