"""Talking to a local Ollama server.

Ollama is a separate program the user installs and runs; this client assumes it
may be missing, asleep, or holding a different model than the one configured,
and says so clearly in each case rather than raising a connection error.

JSON replies get a tolerant parser (`extract_json_object`) because small models
like to wrap their output in prose or markdown fences even when told not to.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from app.errors import OllamaError
from app.logging_setup import get_logger

log = get_logger(__name__)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
CONNECT_TIMEOUT = 5.0
MAX_ATTEMPTS = 3


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply.

    Handles: clean JSON, markdown-fenced JSON, and JSON with commentary before
    or after it. Raises OllamaError if there is nothing object-shaped in there.
    """
    if not text or not text.strip():
        raise OllamaError("The model returned an empty reply.", "")

    candidates: list[str] = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text.strip())

    # Last resort: the outermost {...} span in the reply.
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first:last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}

    raise OllamaError(
        "The model did not return valid JSON.",
        "Try a different OLLAMA_MODEL; qwen2.5:7b-instruct follows JSON "
        "instructions more reliably than very small models.",
    )


class OllamaClient:
    """A small, synchronous Ollama client with health checks and retries."""

    def __init__(self, host: str = "http://127.0.0.1:11434",
                 model: str = "qwen2.5:7b-instruct",
                 timeout_seconds: int = 600, num_ctx: int = 8192) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_seconds = int(timeout_seconds)
        self.num_ctx = int(num_ctx)

    # --- health -----------------------------------------------------------

    def list_models(self) -> list[str]:
        """Names of the models Ollama currently has pulled."""
        try:
            with httpx.Client(timeout=CONNECT_TIMEOUT) as client:
                response = client.get(f"{self.host}/api/tags")
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.host} ({exc.__class__.__name__}).",
                "Start it with 'ollama serve', or install it from https://ollama.com "
                "if it is not on this machine yet.",
            ) from exc
        except ValueError as exc:
            raise OllamaError(
                f"Ollama at {self.host} returned a reply this app could not read.",
                "Check that OLLAMA_HOST points at Ollama and not another service.",
            ) from exc

        return [str(item.get("name", "")) for item in payload.get("models", [])]

    def health(self) -> dict[str, Any]:
        """Report whether Ollama is usable, without raising."""
        try:
            models = self.list_models()
        except OllamaError as exc:
            return {"ok": False, "model": self.model, "models": [],
                    "error": exc.message, "remedy": exc.remedy}

        if not self._model_present(models):
            return {
                "ok": False, "model": self.model, "models": models,
                "error": f"Ollama is running but does not have {self.model!r}.",
                "remedy": f"Pull it once with: ollama pull {self.model}",
            }
        return {"ok": True, "model": self.model, "models": models,
                "error": "", "remedy": ""}

    def _model_present(self, models: list[str]) -> bool:
        # Ollama reports "qwen2.5:7b-instruct"; a user may configure "qwen2.5".
        wanted = self.model.split(":")[0].casefold()
        return any(name.casefold() == self.model.casefold()
                   or name.split(":")[0].casefold() == wanted for name in models)

    def ensure_ready(self) -> None:
        """Raise a helpful OllamaError unless the configured model is usable."""
        status = self.health()
        if not status["ok"]:
            raise OllamaError(str(status["error"]), str(status["remedy"]))

    # --- generation -------------------------------------------------------

    def chat(self, system: str, user: str, *, json_mode: bool = True,
             temperature: float = 0.2, max_tokens: int | None = None) -> str:
        """Send one chat turn and return the reply text."""
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                # Repetition is the main failure mode of small models on long
                # transcripts; a mild penalty keeps replies from looping.
                "repeat_penalty": 1.1,
            },
        }
        if json_mode:
            payload["format"] = "json"
        if max_tokens:
            payload["options"]["num_predict"] = int(max_tokens)

        timeout = httpx.Timeout(self.timeout_seconds, connect=CONNECT_TIMEOUT)
        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(f"{self.host}/api/chat", json=payload)
                    response.raise_for_status()
                    body = response.json()
            except httpx.TimeoutException as exc:
                last_error = exc
                log.warning("Ollama timed out (attempt %d/%d)", attempt, MAX_ATTEMPTS)
            except httpx.HTTPStatusError as exc:
                # A bad request will fail the same way every time; do not retry.
                raise OllamaError(
                    f"Ollama rejected the request (HTTP {exc.response.status_code}).",
                    f"Check that the model {self.model!r} is pulled: "
                    f"ollama pull {self.model}",
                ) from exc
            except httpx.HTTPError as exc:
                last_error = exc
                log.warning("Ollama request failed (attempt %d/%d): %s",
                            attempt, MAX_ATTEMPTS, exc.__class__.__name__)
            except ValueError as exc:
                last_error = exc
                log.warning("Ollama returned unreadable JSON (attempt %d/%d)",
                            attempt, MAX_ATTEMPTS)
            else:
                content = str(body.get("message", {}).get("content", "")).strip()
                if content:
                    return content
                last_error = OllamaError("empty reply", "")
                log.warning("Ollama returned an empty reply (attempt %d/%d)",
                            attempt, MAX_ATTEMPTS)

            if attempt < MAX_ATTEMPTS:
                time.sleep(min(8.0, 2.0 ** attempt))

        raise OllamaError(
            f"Ollama did not answer after {MAX_ATTEMPTS} attempts "
            f"({last_error.__class__.__name__ if last_error else 'unknown error'}).",
            "Check that 'ollama serve' is running and that the machine is not out "
            "of memory. A smaller model such as llama3.2:3b needs far less RAM.",
        )

    def chat_json(self, system: str, user: str, *, temperature: float = 0.2,
                  max_tokens: int | None = None) -> dict[str, Any]:
        """Send one chat turn and parse the reply as a JSON object."""
        reply = self.chat(system, user, json_mode=True, temperature=temperature,
                          max_tokens=max_tokens)
        return extract_json_object(reply)
