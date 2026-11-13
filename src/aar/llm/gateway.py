"""A small chat client for the local model gateway (CLIProxyAPI): OpenAI-compatible
``POST /v1/chat/completions`` and ``GET /v1/models`` with ``Authorization: Bearer <key>``.

**Configuration.** ``gateway_from_env`` reads ``AAR_LLM_BASE_URL`` (default
``http://127.0.0.1:8317``) and ``AAR_LLM_API_KEY`` (required; there is no flag for the key). The
key must be an RFC 6750 bearer token (letters, digits, ``-._~+/``, then ``=`` padding), so it
can never break a header line.

**Safety**, the rules of ``aar.analyze.gemini`` through the shared ``aar.httpsafe``:

- The key is sent only in the ``Authorization`` header, and only to the base URL's own scheme,
  host and port. A base that is not loopback must use https. A base may not carry a user name,
  a query, a fragment or the key.
- No redirect is followed (every 3xx is the error ``redirect``), and proxy environment
  variables are ignored. An opener handed in must be as safe as ``httpsafe.safe_opener()``.
- The key, raw and URL-encoded, is scrubbed from every error message and hint, every log line,
  the reply's text and model, and the client's repr. Errors are raised outside the ``except``
  block that caught a transport failure, so no exception chain carries the request.
- One attempt gets ``timeout_s``: it is the socket timeout, and also a deadline on ``clock`` for
  the whole reply, which is read one socket read at a time, so a server that trickles bytes
  cannot hold it open. The reply body is capped at 1 MiB.
- 429 and 503 are retried at most twice, waiting ``Retry-After`` seconds bounded to 30 (1 s when
  absent or not a number). Nothing else is retried.

**The reply** is checked here: a JSON object whose ``choices[0].message.content`` is a string
with more than whitespace in it. ``model`` (a string) and ``usage`` (an object whose
``prompt_tokens``, ``completion_tokens`` and ``total_tokens``, when present, are whole numbers)
are optional; ``null`` counts as absent. Anything else is ``bad-response``. ``models()`` lists
the model ids the gateway serves (``{"data": [{"id": ...}, ...]}``; an entry without a valid id
is ignored). The gateway answers a chat for a model it does not serve with HTTP 400, so a caller
that must tell a bad model id from a bad request asks ``models()`` first.

**Errors** are ``GatewayError(code, message, hint)``. The codes are stable: ``not_configured``,
``http-<status>``, ``timeout``, ``network``, ``redirect``, ``too-large``, ``bad-response`` and
``origin``. A bad argument to ``chat`` is a programming error and raises ``ValueError``.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from aar.httpsafe import (
    MAX_RETRIES,
    RETRIED_STATUSES,
    RedirectRefused,
    Scrubber,
    base_url_problem,
    opener_problem,
    origin,
    retry_after_s,
    safe_opener,
)

BASE_ENV = "AAR_LLM_BASE_URL"
KEY_ENV = "AAR_LLM_API_KEY"
DEFAULT_BASE_URL = "http://127.0.0.1:8317"
CHAT_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"
DEFAULT_TIMEOUT_S = 60.0
MAX_BODY = 1024 * 1024
USAGE_COUNTS = ("prompt_tokens", "completion_tokens", "total_tokens")
_CHUNK = 64 * 1024
_ERROR_READ = 2048
_EXCERPT = 300
_TOKEN = re.compile(r"[A-Za-z0-9._~+/-]+=*")  # RFC 6750 b64token
_MODEL_ID = re.compile(r"[!-~]{1,128}")  # printable ASCII, no spaces
_MAX_MODELS = 10_000
_NOT_JSON = object()
_log = logging.getLogger(__name__)


class GatewayError(Exception):
    """The gateway could not be used. ``code`` is one of the stable codes in the module doc."""

    def __init__(self, code: str, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


@dataclass(frozen=True)
class ChatReply:
    """The model's text, and the model id and token counts the gateway reported."""

    text: str
    model: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)


@dataclass
class _Outcome:
    kind: str  # ok, http, redirect, timeout, network, too-large
    status: int = 0
    body: bytes = b""
    retry_after: str | None = None
    reason: str = ""


class GatewayClient:
    """Chat completions through the gateway. The key lives only in this object and in the
    ``Authorization`` header of its requests."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        opener: urllib.request.OpenerDirector | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        key = api_key.strip() if isinstance(api_key, str) else ""
        if not key:
            raise GatewayError(
                "not_configured",
                f"the gateway key ({KEY_ENV}) is empty",
                f"export {KEY_ENV} with the gateway's key",
            )
        if not _TOKEN.fullmatch(key):
            raise GatewayError(
                "not_configured",
                f"the gateway key ({KEY_ENV}) is not a bearer token: only letters, digits, "
                "-._~+/ and = padding at the end are allowed",
                f"check the value of {KEY_ENV}",
            )
        self._key = key
        self.scrub = Scrubber(key)
        base = base_url.strip().rstrip("/") if isinstance(base_url, str) else ""
        problem = self._base_problem(base)
        if problem is not None:
            raise GatewayError(
                "origin",
                f"{BASE_ENV} {problem}",
                f"set {BASE_ENV} to https://host[:port], or http:// on a loopback host",
            )
        if not _is_number(timeout_s) or not timeout_s > 0:
            raise ValueError("timeout_s must be a positive number of seconds")
        if opener is None:
            opener = safe_opener()
        elif (unsafe := opener_problem(opener)) is not None:
            raise ValueError(f"the opener is refused: {unsafe}; use aar.httpsafe.safe_opener()")
        self.base_url = base
        self.timeout_s = float(timeout_s)
        self._origin = origin(base)
        self._opener = opener
        self._sleep = sleep
        self._clock = clock

    def _base_problem(self, base: str) -> str | None:
        try:
            problem = base_url_problem(base)
        except ValueError:  # a port that is not a number in range
            problem = "form"
        if problem == "form":
            return "must be http(s)://host[:port], with no query or fragment"
        if problem == "https":
            return "is not on a loopback host, so it must use https"
        if "@" in urllib.parse.urlsplit(base).netloc:
            return "must not carry a user name or password"
        if self.scrub(base) != base:
            return f"contains the gateway key, which belongs only in {KEY_ENV}"
        return None

    def __repr__(self) -> str:
        return self.scrub(
            f"GatewayClient(base_url={self.base_url!r}, timeout_s={self.timeout_s:g})"
        )

    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        temperature: float | None = None,
    ) -> ChatReply:
        """One chat completion. Raises ``GatewayError``; a bad argument raises ``ValueError``."""
        if not is_model_id(model):
            raise ValueError("model must be 1-128 printable ASCII characters without spaces")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a whole number of at least 1")
        body: dict[str, Any] = {
            "model": model,
            "messages": _plain_messages(messages),
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            if not _is_number(temperature) or not 0 <= temperature <= 2:
                raise ValueError("temperature must be a number from 0 to 2")
            body["temperature"] = temperature
        reply = self._post(self.base_url + CHAT_PATH, json.dumps(body).encode("utf-8"))
        return self._reply(reply.body)

    def models(self) -> tuple[str, ...]:
        """The ids of the models the gateway serves, in its order. Raises ``GatewayError``."""
        reply = self._request("GET", self.base_url + MODELS_PATH, None)
        data = _loads(reply.body)
        listed = data.get("data") if isinstance(data, dict) else None
        if not isinstance(listed, list):
            raise self._bad("the reply is not a JSON object with a data list", f"GET {MODELS_PATH}")
        ids = (entry.get("id") for entry in listed[:_MAX_MODELS] if isinstance(entry, dict))
        return tuple(self.scrub(model) for model in ids if is_model_id(model))

    # ------------------------------------------------------------ transport

    def _error(self, code: str, message: str, hint: str = "") -> GatewayError:
        return GatewayError(code, self.scrub(message), self.scrub(hint))

    def _post(self, url: str, data: bytes) -> _Outcome:
        return self._request("POST", url, data)

    def _request(self, method: str, url: str, data: bytes | None) -> _Outcome:
        if origin(url) != self._origin:
            raise self._error(
                "origin",
                f"refusing a request to {origin(url)[1] or 'an unknown host'}: the key is only "
                "ever sent to the gateway's own scheme, host and port",
            )
        path = self.scrub(urllib.parse.urlsplit(url).path)
        for attempt in range(MAX_RETRIES + 1):
            request = urllib.request.Request(url, data=data, method=method)
            request.add_header("Authorization", f"Bearer {self._key}")
            if data is not None:
                request.add_header("Content-Type", "application/json")
            request.add_header("Accept", "application/json")
            started = self._clock()
            outcome = self._send(request, started + self.timeout_s)
            _log.debug(
                "gateway %s %s: %s (HTTP %s) in %.2f s",
                method,
                path,
                outcome.kind,
                outcome.status or "none",
                self._clock() - started,
            )
            if (
                outcome.kind == "http"
                and outcome.status in RETRIED_STATUSES
                and attempt < MAX_RETRIES
            ):
                wait = retry_after_s(outcome.retry_after)
                _log.warning(
                    "gateway %s %s: HTTP %d, retry %d of %d in %g s",
                    method,
                    path,
                    outcome.status,
                    attempt + 1,
                    MAX_RETRIES,
                    wait,
                )
                self._sleep(wait)
                continue
            break
        if outcome.kind != "ok":
            raise self._failure(f"{method} {path}", outcome)
        return outcome

    def _send(self, request: urllib.request.Request, deadline: float) -> _Outcome:
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                status = response.status
                body, state = self._read(response, deadline, MAX_BODY)
        except RedirectRefused as redirect:
            return _Outcome("redirect", status=redirect.code)
        except urllib.error.HTTPError as exc:
            # Only the status, Retry-After and a short excerpt leave this block; the reason
            # phrase and the response object (which a hostile server can fill) do not.
            excerpt = self._excerpt(exc, deadline)
            retry = exc.headers.get("Retry-After") if exc.headers is not None else None
            exc.close()
            kind = "redirect" if 300 <= exc.code < 400 else "http"
            return _Outcome(kind, status=exc.code, body=excerpt, retry_after=retry)
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
            reason = getattr(exc, "reason", None)
            late = isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError)
            detail = f"{type(exc).__name__}: {reason or exc}"
            return _Outcome("timeout" if late else "network", reason=detail)
        if state == "more":
            return _Outcome("too-large", status=status)
        if state == "late":
            return _Outcome("timeout", status=status)
        return _Outcome("ok", status=status, body=body)

    def _read(self, stream: Any, deadline: float, limit: int) -> tuple[bytes, str]:
        """Up to ``limit + 1`` bytes, one socket read at a time, so a server that trickles
        bytes cannot hold the read past the deadline. The state is ``"ok"`` at the end of the
        stream, ``"more"`` past ``limit`` bytes, ``"late"`` once the deadline has passed."""
        chunks: list[bytes] = []
        size = 0
        while size <= limit:
            chunk = stream.read1(min(_CHUNK, limit + 1 - size))
            if not chunk:
                return b"".join(chunks), "ok"
            chunks.append(chunk)
            size += len(chunk)
            if self._clock() > deadline:
                return b"".join(chunks), "late"
        return b"".join(chunks), "more"

    def _excerpt(self, error: urllib.error.HTTPError, deadline: float) -> bytes:
        """The start of an error reply, or nothing when the read stopped short of the end or
        of ``_ERROR_READ`` bytes: a read cut short may end inside the key, where the scrubber
        cannot see it. (Past ``_ERROR_READ`` bytes a cut is far beyond what ``_failure``
        shows.)"""
        try:
            data, state = self._read(error, deadline, _ERROR_READ)
        except (OSError, ValueError, http.client.HTTPException):
            return b""
        return data if state in ("ok", "more") else b""

    def _failure(self, where: str, outcome: _Outcome) -> GatewayError:
        if outcome.kind == "redirect":
            return self._error(
                "redirect",
                f"{where}: the gateway answered {outcome.status} (a redirect); redirects are "
                "never followed, so the key is never sent anywhere else",
                f"point {BASE_ENV} at the gateway itself",
            )
        if outcome.kind == "too-large":
            return self._error("too-large", f"{where}: the reply is over {MAX_BODY} bytes")
        if outcome.kind == "timeout":
            detail = f" ({outcome.reason})" if outcome.reason else ""
            return self._error(
                "timeout",
                f"{where}: no complete reply within {self.timeout_s:g} s{detail}",
                "the gateway or the model is slow or stalled; nothing was retried",
            )
        if outcome.kind == "network":
            return self._error(
                "network",
                f"{where}: {outcome.reason}",
                f"is the gateway running at {self.base_url}? nothing was retried",
            )
        # Scrubbed, then cut, then whitespace collapsed: a key that straddles the cut is already
        # gone, and collapsing cannot pull a key cut off at _ERROR_READ bytes into the excerpt.
        text = self.scrub(outcome.body.decode("utf-8", errors="replace"))
        excerpt = " ".join(text[:_EXCERPT].split())
        retried = f" after {MAX_RETRIES} retries" if outcome.status in RETRIED_STATUSES else ""
        hint = f"check {KEY_ENV}" if outcome.status in (401, 403) else ""
        return self._error(
            f"http-{outcome.status}", f"{where}: HTTP {outcome.status}{retried}: {excerpt}", hint
        )

    # ------------------------------------------------------------ the reply

    def _reply(self, body: bytes) -> ChatReply:
        data = _loads(body)
        if data is _NOT_JSON:
            raise self._bad("the reply is not JSON")
        if not isinstance(data, dict):
            raise self._bad("the reply is not a JSON object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise self._bad("the reply has no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise self._bad("choices[0] has no message object")
        content = message.get("content")
        if not isinstance(content, str):
            raise self._bad("choices[0].message.content is not a string")
        if not content.strip():
            raise self._bad("choices[0].message.content is empty")
        model = data.get("model")
        if model is not None and not isinstance(model, str):
            raise self._bad("model is not a string")
        usage = data.get("usage")
        if usage is not None and not isinstance(usage, dict):
            raise self._bad("usage is not an object")
        counts: dict[str, int] = {}
        for name in USAGE_COUNTS:
            value = (usage or {}).get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise self._bad(f"usage.{name} is not a whole number")
            counts[name] = value
        return ChatReply(
            text=self.scrub(content),
            model=None if model is None else self.scrub(model),
            usage=counts,
        )

    def _bad(self, message: str, where: str = f"POST {CHAT_PATH}") -> GatewayError:
        return self._error(
            "bad-response",
            f"{where}: {message}",
            f"is {self.base_url} an OpenAI-compatible gateway?",
        )


def is_model_id(value: Any) -> bool:
    """Whether ``value`` can be a model id: 1-128 printable ASCII characters, no spaces."""
    return isinstance(value, str) and _MODEL_ID.fullmatch(value) is not None


def _loads(body: bytes) -> Any:
    """The JSON in ``body``, or ``_NOT_JSON``. ValueError covers bad UTF-8, bad JSON and an
    integer over Python's digit limit, RecursionError a reply nested too deeply; the caller
    raises outside any handler."""
    with contextlib.suppress(ValueError, RecursionError):
        return json.loads(body.decode("utf-8"))
    return _NOT_JSON


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _plain_messages(messages: Any) -> list[dict[str, str]]:
    if isinstance(messages, str | bytes) or not isinstance(messages, Sequence) or not messages:
        raise ValueError("messages must be a non-empty list of {role, content} objects")
    out: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        role = message.get("role") if isinstance(message, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(role, str) or not role or not isinstance(content, str):
            raise ValueError(f"messages[{index}] needs a role and a string content")
        out.append({"role": role, "content": content})
    return out


def gateway_from_env(environ: Mapping[str, str] | None = None) -> GatewayClient:
    """A client for ``AAR_LLM_BASE_URL`` (default ``http://127.0.0.1:8317``) with the key in
    ``AAR_LLM_API_KEY``. A missing or blank key is ``GatewayError("not_configured")`` naming the
    variable; a refused base URL is ``GatewayError("origin")``."""
    env = os.environ if environ is None else environ
    key = env.get(KEY_ENV) or ""
    if not key.strip():
        raise GatewayError(
            "not_configured",
            f"{KEY_ENV} is not set (or blank)",
            f"export {KEY_ENV} with the gateway's key; there is no flag for the key",
        )
    base = (env.get(BASE_ENV) or "").strip() or DEFAULT_BASE_URL
    return GatewayClient(base, key)


__all__ = [
    "BASE_ENV",
    "CHAT_PATH",
    "DEFAULT_BASE_URL",
    "KEY_ENV",
    "MAX_BODY",
    "MODELS_PATH",
    "ChatReply",
    "GatewayClient",
    "GatewayError",
    "gateway_from_env",
    "is_model_id",
]
