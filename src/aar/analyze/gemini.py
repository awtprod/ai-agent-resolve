"""A minimal Gemini REST client for ``analyze.laughs``: upload a file, wait, generate, delete.

**The credential** is an API key read by the caller from ``$AAR_GEMINI_API_KEY`` only. It is
sent only in the ``x-goog-api-key`` header, never in a URL, and every string that crosses this
module's boundary (error messages, reply excerpts) is scrubbed of it. Errors are raised outside
the ``except`` block that caught the transport failure, so no exception chain carries a
response, its headers or its reason phrase.

**No redirects.** urllib's default redirect handler copies request headers, the key included,
to the new location. This opener refuses every 3xx with a typed error, ignores proxy
environment variables, and every request (including the upload URL the server hands back) must
go to the base URL's own scheme, host and port.

**Base URL.** A constructor argument for tests only (no CLI flag, no environment variable). A
non-loopback base must be ``https``.

**Limits.** Reply bodies are capped at 8 MiB. 429 and 503 are retried at most twice, waiting
``Retry-After`` seconds bounded to 30 (1 s when absent or not a number). Every other HTTP error
is a typed error at once.

REST calls (v1beta): resumable upload ``POST upload/v1beta/files`` (``start``, then
``upload, finalize`` at offset 0 on the returned ``x-goog-upload-url``); ``GET v1beta/{name}``
until ``ACTIVE``; ``POST v1beta/models/{model}:generateContent``; ``DELETE v1beta/{name}``.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from aar.analyze.errors import AnalyzeError

KEY_ENV = "AAR_GEMINI_API_KEY"
_GEMINI_BASE = "https://generativelanguage.googleapis.com"
MAX_BODY = 8 * 1024 * 1024
MAX_RETRIES = 2
RETRY_AFTER_MAX_S = 30.0
POLL_S = 2.0
_RETRIED = (429, 503)
_LOOPBACK = ("127.0.0.1", "::1", "localhost")
_FILE_NAME = re.compile(r"files/[A-Za-z0-9_-]{1,128}")
_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_EXCERPT = 300
REDACTED = "[redacted]"
_NOT_JSON = object()


class Scrubber:
    """Removes a secret (raw and URL-encoded) from text."""

    def __init__(self, secret: str) -> None:
        forms = {secret, urllib.parse.quote(secret, safe=""), urllib.parse.quote_plus(secret)}
        self._needles = sorted((f for f in forms if f), key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for needle in self._needles:
            text = text.replace(needle, REDACTED)
        return text


@dataclass(frozen=True)
class FileRef:
    name: str  # "files/<id>"
    uri: str
    mime_type: str
    state: str = ""


class GeminiTransport(Protocol):
    def upload(self, path: Path, mime: str, display_name: str) -> FileRef: ...

    def wait_active(self, ref: FileRef, timeout_s: float) -> FileRef: ...

    def generate(
        self, model: str, ref: FileRef, prompt: str, response_schema: dict[str, Any]
    ) -> dict[str, Any]: ...

    def delete(self, ref: FileRef) -> None: ...


class _Redirect(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: following one would send the key to the new location."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        # (req, fp, code, msg, headers, newurl): only the status code is kept.
        raise _Redirect(args[2] if len(args) > 2 else int(kwargs.get("code", 0)))


@dataclass
class _Outcome:
    kind: str  # ok, http, redirect, network, too-large
    status: int = 0
    headers: Any = None
    body: bytes = b""
    retry_after: str | None = None
    reason: str = ""


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urllib.parse.urlsplit(url)
    return parts.scheme, (parts.hostname or "").lower(), parts.port


def _retry_after(value: str | None) -> float:
    try:
        seconds = float(value) if value is not None else 1.0
    except ValueError:
        seconds = 1.0
    if seconds != seconds:  # NaN
        seconds = 1.0
    return max(0.0, min(RETRY_AFTER_MAX_S, seconds))


class UrllibGeminiTransport:
    """``GeminiTransport`` over urllib. The key lives only in this object and request headers."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise AnalyzeError("missing-credential", f"{KEY_ENV} is empty")
        self._key = api_key.strip()
        self.scrub = Scrubber(self._key)
        base = (base_url if base_url is not None else _GEMINI_BASE).rstrip("/")
        scheme, host, _ = _origin(base)
        parts = urllib.parse.urlsplit(base)
        if scheme not in ("https", "http") or not host or parts.query or parts.fragment:
            raise AnalyzeError("invalid-argument", "the Gemini base URL must be http(s)://host")
        if scheme != "https" and host not in _LOOPBACK:
            raise AnalyzeError(
                "invalid-argument", "a Gemini base URL that is not loopback must use https"
            )
        self.base = base
        self._origin = _origin(base)
        self.timeout_s = timeout_s
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def __repr__(self) -> str:
        return f"UrllibGeminiTransport(base_url={self.base!r})"

    # ------------------------------------------------------------ plumbing

    def _where(self, url: str) -> str:
        return urllib.parse.urlsplit(url).path  # never the query (an upload id)

    def _error(self, kind: str, message: str, hint: str = "") -> AnalyzeError:
        return AnalyzeError(kind, self.scrub(message), hint=self.scrub(hint))

    def _send(self, req: urllib.request.Request) -> _Outcome:
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                body = resp.read(MAX_BODY + 1)
                status, headers = resp.status, resp.headers
        except _Redirect as redirect:
            return _Outcome("redirect", status=redirect.code)
        except urllib.error.HTTPError as exc:
            # Only the status, Retry-After and a short excerpt leave this block; the reason
            # phrase and the response object (which a hostile server can fill) do not.
            try:
                excerpt = exc.read(2048)
            except (OSError, ValueError, http.client.HTTPException):
                excerpt = b""
            retry = exc.headers.get("Retry-After") if exc.headers is not None else None
            exc.close()
            return _Outcome("http", status=exc.code, body=excerpt, retry_after=retry)
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
            reason = getattr(exc, "reason", None)
            return _Outcome("network", reason=f"{type(exc).__name__}: {reason or exc}")
        if len(body) > MAX_BODY:
            return _Outcome("too-large", status=status)
        return _Outcome("ok", status=status, headers=headers, body=body)

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        upload: Path | None = None,
    ) -> _Outcome:
        if _origin(url) != self._origin:
            raise self._error(
                "gemini-bad-reply",
                f"refusing a request to {_origin(url)[1] or 'an unknown host'}: not the API "
                "host (the key is only ever sent to the base URL's host)",
            )
        where = self._where(url)
        for attempt in range(MAX_RETRIES + 1):
            data: Any = body
            handle = None
            if upload is not None:
                handle = open(upload, "rb")  # noqa: SIM115 - closed below, per attempt
                data = handle
            try:
                req = urllib.request.Request(url, data=data, method=method)
                req.add_header("x-goog-api-key", self._key)
                for name, value in (headers or {}).items():
                    req.add_header(name, value)
                outcome = self._send(req)
            finally:
                if handle is not None:
                    handle.close()
            if outcome.kind == "ok":
                return outcome
            if outcome.kind == "http" and outcome.status in _RETRIED and attempt < MAX_RETRIES:
                self._sleep(_retry_after(outcome.retry_after))
                continue
            break
        raise self._failure(method, where, outcome)

    def _failure(self, method: str, where: str, outcome: _Outcome) -> AnalyzeError:
        if outcome.kind == "redirect":
            return self._error(
                "gemini-redirect",
                f"{method} {where}: the server answered {outcome.status} (a redirect); "
                "redirects are refused so the key is never sent anywhere else",
            )
        if outcome.kind == "too-large":
            return self._error(
                "gemini-too-large", f"{method} {where}: the reply is over {MAX_BODY} bytes"
            )
        if outcome.kind == "network":
            return self._error(
                "gemini-unreachable",
                f"{method} {where}: {outcome.reason}",
                hint="check the network; nothing was retried",
            )
        excerpt = " ".join(outcome.body.decode("utf-8", errors="replace").split())[:_EXCERPT]
        retried = " after 2 retries" if outcome.status in _RETRIED else ""
        return self._error(
            "gemini-http", f"{method} {where}: HTTP {outcome.status}{retried}: {excerpt}"
        )

    def _json(self, outcome: _Outcome, what: str) -> dict[str, Any]:
        data: Any = _NOT_JSON
        # Raised below, outside any handler: a JSONDecodeError carries the whole reply text.
        with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError):
            data = json.loads(outcome.body.decode("utf-8"))
        if data is _NOT_JSON:
            raise self._error("gemini-bad-reply", f"the {what} reply is not JSON")
        if not isinstance(data, dict):
            raise self._error("gemini-bad-reply", f"the {what} reply is not a JSON object")
        return data

    def _file(self, data: Any, what: str) -> FileRef:
        if not isinstance(data, dict):
            raise self._error("gemini-bad-reply", f"the {what} reply has no file object")
        name, uri, mime = data.get("name"), data.get("uri"), data.get("mimeType")
        if not (isinstance(name, str) and _FILE_NAME.fullmatch(name)):
            raise self._error("gemini-bad-reply", f"the {what} reply has no valid file name")
        if not (isinstance(uri, str) and uri and isinstance(mime, str)):
            raise self._error("gemini-bad-reply", f"the {what} reply has no file uri or type")
        state = data.get("state")
        return FileRef(
            name=name, uri=uri, mime_type=mime, state=state if isinstance(state, str) else ""
        )

    # ------------------------------------------------------------ the four calls

    def upload(self, path: Path, mime: str, display_name: str) -> FileRef:
        size = path.stat().st_size
        start = self._request(
            "POST",
            f"{self.base}/upload/v1beta/files",
            headers={
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": mime,
                "Content-Type": "application/json",
            },
            body=json.dumps({"file": {"display_name": display_name}}).encode("utf-8"),
        )
        upload_url = start.headers.get("x-goog-upload-url") if start.headers else None
        if not upload_url:
            raise self._error("gemini-bad-reply", "the upload start reply has no upload URL")
        done = self._request(
            "POST",
            upload_url,
            headers={
                "Content-Length": str(size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            upload=path,
        )
        return self._file(self._json(done, "upload").get("file"), "upload")

    def wait_active(self, ref: FileRef, timeout_s: float) -> FileRef:
        deadline = self._clock() + timeout_s
        while True:
            reply = self._request("GET", f"{self.base}/v1beta/{ref.name}")
            current = self._file(self._json(reply, "file status"), "file status")
            if current.state == "ACTIVE":
                return current
            if current.state == "FAILED":
                raise self._error("gemini-file-failed", f"{ref.name}: the upload FAILED to process")
            now = self._clock()
            if now >= deadline:
                raise self._error(
                    "gemini-timeout",
                    f"{ref.name} was not ACTIVE after {timeout_s:g} s (state {current.state!r})",
                )
            self._sleep(min(POLL_S, deadline - now))

    def generate(
        self, model: str, ref: FileRef, prompt: str, response_schema: dict[str, Any]
    ) -> dict[str, Any]:
        if not _MODEL_ID.fullmatch(model):
            raise AnalyzeError("invalid-argument", f"{model!r} is not a model id")
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"file_data": {"mime_type": ref.mime_type, "file_uri": ref.uri}},
                        {"text": prompt},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
            },
        }
        reply = self._request(
            "POST",
            f"{self.base}/v1beta/models/{model}:generateContent",
            headers={"Content-Type": "application/json"},
            body=json.dumps(body).encode("utf-8"),
        )
        return self._json(reply, "generateContent")

    def delete(self, ref: FileRef) -> None:
        if not _FILE_NAME.fullmatch(ref.name):
            raise self._error("gemini-bad-reply", "refusing to delete a malformed file name")
        self._request("DELETE", f"{self.base}/v1beta/{ref.name}")


__all__ = [
    "KEY_ENV",
    "MAX_BODY",
    "REDACTED",
    "FileRef",
    "GeminiTransport",
    "Scrubber",
    "UrllibGeminiTransport",
]
