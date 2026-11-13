"""Safety helpers shared by the stdlib HTTP clients that send an API key (``aar.analyze.gemini``
and ``aar.llm.gateway``).

- ``Scrubber`` removes the key from any text that leaves a client: raw, and URL-encoded the
  three ways Python's ``urllib.parse`` writes it (``quote(key, safe="")``, ``quote(key)``, which
  keeps ``/``, and ``quote_plus(key)``), each ``%XX`` escape matched with its hex digits in
  either case.
- ``safe_opener()`` builds a urllib opener that ignores proxy environment variables and refuses
  every redirect: urllib's default handler copies request headers, the key included, to the new
  location. A refused redirect raises ``RedirectRefused``. ``opener_problem()`` checks that an
  opener handed in from outside is as safe.
- ``origin()`` and ``base_url_problem()`` pin requests to the base URL's scheme, host and port,
  and require https unless the host is loopback.
- ``send_with_retries()`` runs the attempts of one request, each classified as an ``Outcome``:
  ``RETRIED_STATUSES`` (429 and 503) are retried at most ``MAX_RETRIES`` times, after
  ``retry_after_s()`` (``Retry-After``, bounded). Each client keeps its own ``_send``, which
  turns one attempt into an ``Outcome``.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

REDACTED = "[redacted]"
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
RETRIED_STATUSES = (429, 503)
MAX_RETRIES = 2
RETRY_AFTER_MAX_S = 30.0
_RETRY_AFTER_DEFAULT_S = 1.0
_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")


class Scrubber:
    """Removes a secret from text: raw, ``quote(secret, safe="")``, ``quote(secret)`` and
    ``quote_plus(secret)``. A ``%XX`` escape matches with its hex digits in either case
    (``%2B`` and ``%2b`` are the same byte); everything else matches exactly."""

    def __init__(self, secret: str) -> None:
        forms = {
            secret,
            urllib.parse.quote(secret, safe=""),
            urllib.parse.quote(secret),
            urllib.parse.quote_plus(secret),
        }
        needles = sorted((form for form in forms if form), key=len, reverse=True)
        self._pattern = re.compile("|".join(map(_needle, needles))) if needles else None

    def __call__(self, text: str) -> str:
        return text if self._pattern is None else self._pattern.sub(REDACTED, text)


def _needle(form: str) -> str:
    """A regex for ``form``: literal, but with each ``%XX`` escape's hex digits in either case."""
    parts: list[str] = []
    end = 0
    for escape in _ESCAPE.finditer(form):
        parts.append(re.escape(form[end : escape.start()]))
        parts.append("%" + "".join(f"[{c.upper()}{c.lower()}]" for c in escape.group()[1:]))
        end = escape.end()
    parts.append(re.escape(form[end:]))
    return "".join(parts)


class RedirectRefused(Exception):
    """The server answered a redirect, which ``NoRedirectHandler`` never follows."""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: following one would send the key to the new location."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        # (req, fp, code, msg, headers, newurl): only the status code is kept.
        raise RedirectRefused(args[2] if len(args) > 2 else int(kwargs.get("code", 0)))


def safe_opener() -> urllib.request.OpenerDirector:
    """A urllib opener that ignores proxy environment variables and refuses every redirect."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirectHandler())


def opener_problem(opener: Any) -> str | None:
    """Why ``opener`` is less safe than ``safe_opener()``, or ``None``. It must be a urllib
    ``OpenerDirector`` whose redirect handlers are all ``NoRedirectHandler`` and whose proxy
    handlers have no proxy."""
    if not isinstance(opener, urllib.request.OpenerDirector):
        return "it is not a urllib OpenerDirector"
    redirects = [h for h in opener.handlers if isinstance(h, urllib.request.HTTPRedirectHandler)]
    if not redirects or not all(isinstance(h, NoRedirectHandler) for h in redirects):
        return "it may follow redirects"
    if any(isinstance(h, urllib.request.ProxyHandler) and h.proxies for h in opener.handlers):
        return "it uses a proxy"
    return None


def origin(url: str) -> tuple[str, str, int | None]:
    """``(scheme, lower-case host, port)``: what a request is pinned to. A bad port raises
    ``ValueError``."""
    parts = urllib.parse.urlsplit(url)
    return parts.scheme, (parts.hostname or "").lower(), parts.port


def base_url_problem(base: str) -> str | None:
    """Why ``base`` cannot be an API base URL, or ``None``: ``"form"`` when it is not
    ``http(s)://host`` without a query or fragment, ``"https"`` when it is plain http to a host
    that is not loopback. A bad port raises ``ValueError``."""
    scheme, host, _ = origin(base)
    parts = urllib.parse.urlsplit(base)
    if scheme not in ("https", "http") or not host or parts.query or parts.fragment:
        return "form"
    if scheme != "https" and host not in LOOPBACK_HOSTS:
        return "https"
    return None


@dataclass
class Outcome:
    """One attempt of a request, as a client's ``_send`` classifies it. ``kind`` is ``ok``,
    ``http`` (an error status, with a short excerpt of its body in ``body``), ``redirect``,
    ``timeout``, ``network`` or ``too-large``; ``headers`` and ``body`` of an ``ok`` reply are
    the reply's own."""

    kind: str
    status: int = 0
    headers: Any = None
    body: bytes = b""
    retry_after: str | None = None
    reason: str = ""


def send_with_retries(
    attempt: Callable[[], Outcome],
    sleep: Callable[[float], None],
    on_retry: Callable[[Outcome, int, float], None] | None = None,
) -> Outcome:
    """Run ``attempt`` once, then again after each ``http`` outcome whose status is in
    ``RETRIED_STATUSES``, at most ``MAX_RETRIES`` more times, sleeping
    ``retry_after_s(outcome.retry_after)`` before each retry (``on_retry(outcome, number,
    seconds)`` hears of it first). Returns the last outcome: ``ok``, or the failure to report."""
    retries = 0
    while True:
        outcome = attempt()
        if (
            outcome.kind != "http"
            or outcome.status not in RETRIED_STATUSES
            or retries >= MAX_RETRIES
        ):
            return outcome
        retries += 1
        wait = retry_after_s(outcome.retry_after)
        if on_retry is not None:
            on_retry(outcome, retries, wait)
        sleep(wait)


def retry_after_s(value: str | None) -> float:
    """Seconds to wait before a retry: ``Retry-After`` as a number of seconds, bounded to
    0..``RETRY_AFTER_MAX_S``; 1 s when it is absent or not a number (an HTTP date is not read)."""
    try:
        seconds = float(value) if value is not None else _RETRY_AFTER_DEFAULT_S
    except ValueError:
        seconds = _RETRY_AFTER_DEFAULT_S
    if seconds != seconds:  # NaN
        seconds = _RETRY_AFTER_DEFAULT_S
    return max(0.0, min(RETRY_AFTER_MAX_S, seconds))


__all__ = [
    "LOOPBACK_HOSTS",
    "MAX_RETRIES",
    "REDACTED",
    "RETRIED_STATUSES",
    "RETRY_AFTER_MAX_S",
    "NoRedirectHandler",
    "Outcome",
    "RedirectRefused",
    "Scrubber",
    "base_url_problem",
    "opener_problem",
    "origin",
    "retry_after_s",
    "safe_opener",
    "send_with_retries",
]
