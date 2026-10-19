"""The human supervision console: review queue, lease board, approvals, emergency stop.

DESIGN R5 gives humans four things: a review queue (render plus timecoded notes),
approve or reject, a board of who holds what, and an emergency stop. This is that
console, plus the approval tokens of DESIGN section 5. It is a small Starlette app
that renders HTML on the server, with inline CSS and a few lines of inline script
(seek the player to a note), and works at phone width.

Identity is network reachability (read this before exposing it). There are no
accounts, passwords or sessions. Whoever can reach the port acts as ``--operator``;
with ``--trust-tailscale-headers``, any process on this host can claim any login by
sending the header to loopback. That is acceptable only on the tailnet with trusted
local processes (DESIGN D2); a real credential is an open item (DESIGN section 7)
and is required before any exposure beyond the tailnet.

Security model (the hub is tailnet-only for v1, DESIGN D2):

- It binds to 127.0.0.1 by default; the intended remote path is ``tailscale serve``
  proxying to loopback. The ``Host`` header must be a loopback name or one passed
  with ``--allowed-host``, which blocks DNS rebinding.
- The human is ``--operator NAME``, or with ``--trust-tailscale-headers`` the
  ``Tailscale-User-Login`` header, honoured only when the TCP peer is loopback (that
  is, the request came through ``tailscale serve``). Any local process can also send
  that header, so only enable it on a host where local processes are trusted. With no
  identity the console is read-only.
- Every write is a POST with a CSRF token bound to the identity and an
  ``Origin``/``Referer`` that must match ``Host``. No GET mutates anything.
- Every string that reaches a page goes through :func:`esc` (agents can write titles,
  notes and reasons, so they can write ``<script>``), and pages carry a nonce-based
  Content-Security-Policy as a second line of defence.
- Artifacts are served only by a validated 64-hex id, never by a path from the
  request, with byte ranges so video can seek, and with ``nosniff`` plus a sandbox
  CSP so an agent-uploaded HTML or SVG file cannot run as the console's origin.
- Every form carries an idempotency key, so a double submit acts once.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import ipaddress
import logging
import os
import re
import secrets
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import BinaryIO
from urllib.parse import parse_qsl, quote, urlsplit

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from aar.hub import approvals, control, reviews
from aar.hub.artifacts import ArtifactStore, CorruptArtifact
from aar.hub.common import (
    MAX_SKIPPED_LISTED,
    Clock,
    HumanRequired,
    InvalidInput,
    SkippedEvent,
    parse_ts,
    utcnow,
)
from aar.hub.events import Actor, EventLog, IdempotencyConflict
from aar.hub.leaseboard import LeaseBoard, lease_board
from aar.hub.registry import Registry

logger = logging.getLogger("aar.hub.console")

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_BODY = 64 * 1024
_CHUNK = 64 * 1024

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@+-]{0,127}$")
_ARTIFACT_RE = re.compile(r"^[0-9a-f]{64}$")
_IDEM_RE = re.compile(r"^[0-9a-f]{32}$")
# ASCII digits only: str.isdigit() accepts "²" and other digits int() rejects.
_UINT_RE = re.compile(r"^[0-9]{1,12}$")
_RANGE_RE = re.compile(r"^bytes=(\d{0,18})-(\d{0,18})$")
_PORT_RE = re.compile(r"^:\d{1,5}$")

# Media types the console will hand to the browser as themselves. Anything else is
# served as a download, so an agent cannot get HTML or SVG rendered as this origin.
_INLINE_MIME = frozenset(
    {
        "video/mp4",
        "video/webm",
        "video/quicktime",
        "audio/mpeg",
        "audio/mp4",
        "audio/wav",
        "audio/x-wav",
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "text/plain",
        "application/json",
    }
)

_TTL_CHOICES = ((900, "15 minutes"), (3600, "1 hour"), (4 * 3600, "4 hours"), (86400, "24 hours"))

# Flash messages are chosen by a fixed code, so nothing from the URL is reflected.
_FLASH = {
    "approved": "Review approved.",
    "rejected": "Review rejected.",
    "noted": "Note added.",
    "granted": "Approval token granted. It is at the top of the list.",
    "revoked": "Token revoked.",
    "stopped": "Emergency stop engaged.",
    "resumed": "Resumed.",
}


# -- HTML: escaping is the default ------------------------------------------


class Html(str):
    """Markup the console built itself. Plain ``str`` values are always escaped."""

    __slots__ = ()


def esc(value: object) -> Html:
    """Escape ``value`` for HTML text or a quoted attribute, unless it is already :class:`Html`."""
    if isinstance(value, Html):
        return value
    return Html(html.escape("" if value is None else str(value), quote=True))


def fmt(template: str, **values: object) -> Html:
    """Fill a trusted template; every value is escaped unless it is :class:`Html`."""
    return Html(template.format_map({key: esc(value) for key, value in values.items()}))


def join(parts: list[Html] | Iterator[Html]) -> Html:
    """Concatenate fragments (each escaped unless already :class:`Html`)."""
    return Html("".join(esc(part) for part in parts))


def url(*segments: str, **query: str) -> str:
    """Build a same-origin path; each segment is percent-encoded so ids cannot add paths."""
    path = "/" + "/".join(quote(segment, safe="") for segment in segments)
    if query:
        path += "?" + "&".join(f"{quote(k, safe='')}={quote(v, safe='')}" for k, v in query.items())
    return path


# -- config, identity, CSRF ---------------------------------------------------


def validate_operator(name: str) -> str:
    """Return ``name`` if it is a usable operator id, else raise :class:`InvalidInput`."""
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise InvalidInput(
            "operator",
            "must be 1-128 characters of letters, digits and . _ @ + -, starting with a "
            "letter or digit",
        )
    return name


@dataclass(frozen=True)
class ConsoleConfig:
    """How the console identifies its human and which ``Host`` names it answers to."""

    operator: str | None = None
    trust_tailscale_headers: bool = False
    allowed_hosts: tuple[str, ...] = ()
    csrf_secret: bytes = field(default_factory=lambda: secrets.token_bytes(32), repr=False)
    csrf_max_age: float = 12 * 3600.0

    def __post_init__(self) -> None:
        if self.operator is not None:
            validate_operator(self.operator)
        if len(self.csrf_secret) < 16:
            raise ValueError("csrf_secret must be at least 16 bytes")


class CsrfSigner:
    """Stateless CSRF tokens: an HMAC over the issue time and the human's identity."""

    def __init__(
        self, secret: bytes, *, max_age: float, now: Callable[[], float] = time.time
    ) -> None:
        self._secret = secret
        self._max_age = max_age
        self._now = now

    def _mac(self, issued: int, identity: str) -> str:
        message = f"{issued}|{identity}".encode()
        return hmac.new(self._secret, message, hashlib.sha256).hexdigest()

    def issue(self, identity: str) -> str:
        issued = int(self._now())
        return f"{issued}.{self._mac(issued, identity)}"

    def verify(self, token: str, identity: str) -> bool:
        issued_text, sep, mac = token.partition(".")
        if not sep or not issued_text.isdigit() or len(issued_text) > 12:
            return False
        issued = int(issued_text)
        age = self._now() - issued
        if age < -60 or age > self._max_age:
            return False
        return hmac.compare_digest(mac, self._mac(issued, identity))


def _host_name(value: str | None) -> str | None:
    """Return the lowercased host name of a ``Host`` header (port and brackets removed)."""
    if not value or len(value) > 255:
        return None
    value = value.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return None
        name, rest = value[1:end], value[end + 1 :]
    else:
        name, colon, port = value.partition(":")
        rest = colon + port
    if rest and not _PORT_RE.fullmatch(rest):
        return None
    return name or None


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return address.is_loopback or (mapped is not None and mapped.is_loopback)


_CONTENT_LENGTH_RE = re.compile(r"^[0-9]{1,18}$")
MAX_FIELDS = 32


async def read_form(request: Request) -> FormData:
    """Read a urlencoded form, enforcing the body cap while streaming.

    The cap is enforced here, not by Starlette's ``max_body_size``: that one replaces
    whatever the app answers with a bare text 413, losing the structured error and the
    security headers. A declared ``Content-Length`` over the cap is refused before any
    byte is read; a body without one is counted as it arrives.
    """
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type != "application/x-www-form-urlencoded":
        raise ConsoleError(
            415, "unsupported_media_type", "console forms are application/x-www-form-urlencoded"
        )
    declared = request.headers.get("content-length")
    if declared is not None:
        if not _CONTENT_LENGTH_RE.fullmatch(declared.strip()):
            raise ConsoleError(400, "bad_content_length", "Content-Length is not a number")
        if int(declared) > MAX_BODY:
            raise _too_large()
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY:
            raise _too_large()
        chunks.append(chunk)
    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConsoleError(400, "bad_encoding", "the form is not valid UTF-8") from exc
    try:
        pairs = parse_qsl(text, keep_blank_values=True, max_num_fields=MAX_FIELDS)
    except ValueError as exc:
        raise ConsoleError(
            400, "too_many_fields", f"a form has at most {MAX_FIELDS} fields"
        ) from exc
    return FormData(pairs)


def _too_large() -> ConsoleError:
    return ConsoleError(
        413,
        "body_too_large",
        f"the form is larger than {MAX_BODY // 1024} KiB",
        "Shorten the text and submit again.",
    )


_DEFAULT_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"cache-control", b"no-store"),
    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),
    (b"referrer-policy", b"same-origin"),
)


class SecurityHeaders:
    """Add the console's security headers to every response that lacks them.

    Pages set their own (a nonce CSP); this catches everything else: JSON errors,
    the Host guard's 421 and anything a library answers directly.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                headers += [(n, v) for n, v in _DEFAULT_HEADERS if n not in present]
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class HostGuard:
    """Refuse any request whose ``Host`` is not an allowed name (blocks DNS rebinding)."""

    def __init__(self, app: ASGIApp, allowed: frozenset[str]) -> None:
        self.app = app
        self.allowed = allowed

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            raw = dict(scope.get("headers") or []).get(b"host", b"").decode("latin-1")
            name = _host_name(raw)
            if name is None or name not in self.allowed:
                shown = repr(raw[:100])
                response = PlainTextResponse(
                    f"Host {shown} is not served here. If this is the name you reach the "
                    "console by, restart it with --allowed-host <name>.",
                    status_code=421,
                    headers={"X-Content-Type-Options": "nosniff"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


# -- errors ---------------------------------------------------------------------


class ConsoleError(Exception):
    """A structured, human-readable failure with an HTTP status."""

    def __init__(self, status: int, code: str, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.hint = hint


def _map_error(exc: BaseException) -> ConsoleError | None:
    """Translate a domain error into a :class:`ConsoleError`; ``None`` means unexpected."""
    if isinstance(exc, ConsoleError):
        return exc
    if isinstance(exc, HumanRequired):
        return ConsoleError(403, "human_required", str(exc))
    if isinstance(exc, reviews.ReviewAlreadyDecided):
        return ConsoleError(409, "already_decided", str(exc), "Reload to see the decision.")
    if isinstance(exc, reviews.ReviewNotOpen):
        return ConsoleError(409, "review_not_open", str(exc))
    if isinstance(exc, reviews.SubjectCancelled):
        return ConsoleError(409, "subject_cancelled", str(exc))
    if isinstance(exc, reviews.ReviewNotFound):
        return ConsoleError(404, "review_not_found", str(exc))
    if isinstance(exc, approvals.ApprovalNotFound):
        return ConsoleError(404, "token_not_found", str(exc))
    if isinstance(exc, approvals.ApprovalError):
        return ConsoleError(409, f"token_{exc.code}", str(exc), "Reload to see its state.")
    if isinstance(exc, control.NotStopped):
        return ConsoleError(409, "not_stopped", str(exc), "Reload: someone may have resumed it.")
    if isinstance(exc, IdempotencyConflict):
        return ConsoleError(
            409, "form_reused", str(exc), "Reload the page and submit the fresh form."
        )
    if isinstance(exc, InvalidInput):
        return ConsoleError(422, "invalid_input", str(exc))
    return None


# -- the app --------------------------------------------------------------------


_CSS = """
:root{--bg:#f4f3ef;--fg:#1a1a1a;--muted:#5b5b5b;--line:#d8d4cb;--card:#fff;--ink:#16181d;
--ok:#17663a;--bad:#b3261e;--info:#1d4ed8}
*{box-sizing:border-box}
body{margin:0;font:16px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
background:var(--bg);color:var(--fg)}
a{color:var(--info)}
header{background:var(--ink);color:#fff;padding:.55rem 1rem;display:flex;flex-wrap:wrap;
gap:.4rem 1rem;align-items:center}
.brand{font-weight:800;letter-spacing:.03em;margin-right:.5rem}
nav{display:flex;flex-wrap:wrap;gap:.25rem}
nav a{color:#d6d9e0;text-decoration:none;padding:.35rem .6rem;border-radius:6px}
nav a[aria-current=page]{background:#30364a;color:#fff}
nav a.stop{background:var(--bad);color:#fff;font-weight:800;letter-spacing:.05em}
.who{margin-left:auto;font-size:.85rem;color:#b9bec9}
.banner{background:var(--bad);color:#fff;padding:.7rem 1rem;font-weight:600}
.banner a{color:#fff}
main{max-width:1120px;margin:0 auto;padding:1rem}
h1{font-size:1.35rem;margin:.25rem 0 1rem}
h2{font-size:1.1rem;margin:0 0 .35rem;overflow-wrap:anywhere}
h3{font-size:1rem;margin:1rem 0 .4rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1rem;
margin:0 0 1rem}
.meta{color:var(--muted);font-size:.85rem;overflow-wrap:anywhere}
.mono,code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.85em}
.mono{overflow-wrap:anywhere}
code{white-space:nowrap}
video,img.render{display:block;width:100%;max-height:62vh;background:#000;border-radius:6px;
margin:.6rem 0}
.renders{display:flex;flex-wrap:wrap;gap:.5rem;font-size:.85rem}
.review-body{display:grid;gap:.25rem 1.25rem;align-items:start}
.review-body h3:first-child{margin-top:.6rem}
.subform{border-top:1px solid var(--line);margin-top:1rem;padding-top:.25rem}
@media (min-width:900px){.review-body{grid-template-columns:minmax(0,3fr) minmax(0,2fr)}
.review-body .side{grid-column:2;grid-row:1/span 2}.review-body .media,.review-body .subform{
grid-column:1}}
ol.notes{list-style:none;padding:0;margin:.4rem 0}
ol.notes li{display:flex;gap:.6rem;align-items:flex-start;padding:.45rem 0;
border-top:1px solid var(--line)}
.note-text{flex:1;white-space:pre-wrap;overflow-wrap:anywhere}
button.tc{font:600 .8rem ui-monospace,Menlo,monospace;background:#eef1f8;color:#1b2c5c;
border:1px solid #c9d2e8;border-radius:6px;padding:.4rem .55rem;min-height:2.25rem;
white-space:nowrap}
button.tc[disabled]{opacity:.75}
label{display:block;font-weight:600;font-size:.9rem;margin:.6rem 0 .25rem}
textarea,input,select{font:inherit;width:100%;padding:.5rem .6rem;border:1px solid #bdb8ad;
border-radius:6px;background:#fff}
textarea{min-height:4.5rem}
.row{display:flex;flex-wrap:wrap;gap:.6rem}
.row>*{flex:1 1 12rem}
.actions{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:.6rem}
button{font:inherit;font-weight:600;padding:.6rem 1.1rem;border-radius:8px;
border:1px solid transparent;cursor:pointer}
button.approve{background:var(--ok);color:#fff}
button.reject,button.danger{background:var(--bad);color:#fff}
button.plain{background:#e8e5dd;color:var(--fg);border-color:#cfc9bc}
.big-stop{display:block;width:100%;font-size:2.1rem;font-weight:900;letter-spacing:.12em;
padding:1.4rem;border-radius:14px;background:var(--bad);color:#fff;
box-shadow:0 6px 0 #7d1a15}
.badge{display:inline-block;white-space:nowrap;padding:.05rem .5rem;border-radius:999px;
font-size:.72rem;font-weight:800;text-transform:uppercase;letter-spacing:.05em;vertical-align:middle;
background:#e7e4dc;color:#3d3a33}
.badge.active,.badge.pending{background:#dbe6ff;color:#163a8a}
.badge.done,.badge.approve{background:#d9f0e0;color:#145c2b}
.badge.failed,.badge.reject,.badge.revoked{background:#fbe0dd;color:#8c1d16}
.badge.consumed,.badge.expired,.badge.cancelled{background:#e7e4dc;color:#55524b}
.badge.human{background:#fff1c7;color:#6b4b00}
.badge.agent{background:#e3e0ff;color:#33268a}
.flash{background:#e3f4e7;border:1px solid #b3dcbf;padding:.6rem .8rem;border-radius:8px;
margin:0 0 1rem}
.error{background:#fdecea;border:1px solid #f0b4ae;padding:.8rem 1rem;border-radius:8px;
margin:0 0 1rem}
.decision{border-left:4px solid var(--line);padding:.3rem .8rem;margin:.6rem 0;
background:#faf9f6}
.decision.approve{border-color:var(--ok)}
.decision.reject{border-color:var(--bad)}
.empty{color:var(--muted);font-style:italic}
.skipped{background:#fff6dc;border:1px solid #e6cf85;border-radius:8px;padding:.5rem .8rem;
margin:0 0 1rem;font-size:.9rem;overflow-wrap:anywhere}
.skipped summary{cursor:pointer;font-weight:600}
.stopped{background:#fdecea;border:2px solid var(--bad)}
table.board{width:100%;border-collapse:collapse;font-size:.92rem}
table.board th,table.board td{text-align:left;padding:.45rem .5rem;
border-bottom:1px solid var(--line);vertical-align:top;overflow-wrap:break-word}
table.board th{font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
@media (prefers-color-scheme:dark){
:root{--bg:#121316;--fg:#e8e6e1;--muted:#a6a49e;--line:#34363c;--card:#1b1d22;--info:#8ab4ff}
textarea,input,select{background:#121316;color:var(--fg);border-color:#4a4d55}
button.plain{background:#2a2d34;color:var(--fg);border-color:#4a4d55}
button.tc{background:#1f2a44;color:#cfe0ff;border-color:#34466f}
.decision{background:#16181c}
.flash{background:#14301f;border-color:#235c37;color:#cfeedb}
.error{background:#3a1715;border-color:#6e2a25}
.stopped{background:#2a1413}
.skipped{background:#2e2610;border-color:#5c4b1c}
.badge{background:#2f3137;color:#d4d2cc}
.badge.active,.badge.pending{background:#1e3366;color:#cfe0ff}
.badge.done,.badge.approve{background:#173d26;color:#bde8cb}
.badge.failed,.badge.reject,.badge.revoked{background:#4a1c19;color:#f6c4bf}
.badge.consumed,.badge.expired,.badge.cancelled{background:#2f3137;color:#b8b6b0}
.badge.human{background:#4a3a0c;color:#ffe08a}
.badge.agent{background:#2c2660;color:#d6d0ff}
}
@media (max-width:640px){
main{padding:.75rem}
nav a{padding:.5rem .7rem}
.who{margin-left:0;width:100%}
table.board thead{display:none}
table.board tr{display:block;border:1px solid var(--line);border-radius:8px;margin:0 0 .6rem;
background:var(--card)}
table.board td{display:flex;gap:.6rem;border:0;padding:.3rem .6rem;overflow-wrap:anywhere}
table.board td::before{content:attr(data-label);flex:0 0 6.5rem;font-weight:600;
color:var(--muted)}
.big-stop{font-size:1.7rem}
}
"""

_SCRIPT = """
document.addEventListener('click', function (event) {
  var seek = event.target.closest('[data-seek]');
  if (seek) {
    var player = document.getElementById(seek.dataset.player);
    if (player) { player.currentTime = parseFloat(seek.dataset.seek); player.focus(); }
  }
  var grab = event.target.closest('[data-frame-from]');
  if (grab) {
    var video = document.getElementById(grab.dataset.frameFrom);
    var input = document.getElementById(grab.dataset.input);
    if (video && input) {
      input.value = Math.floor(video.currentTime * parseFloat(grab.dataset.fps) + 1e-6);
    }
  }
});
"""


def _when(ts: str | None, *, short: bool = False) -> str:
    if not ts:
        return "-"
    try:
        return parse_ts(ts).strftime("%m-%d %H:%M UTC" if short else "%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return ts


def _badge(text: str, kind: str | None = None) -> Html:
    return fmt('<span class="badge {kind}">{text}</span>', kind=kind or text, text=text)


def _wants_json(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "application/json" in accept and "text/html" not in accept


@dataclass
class _Page:
    """Per-request rendering context."""

    request: Request
    actor: Actor | None
    csrf: str | None
    nonce: str
    players: int = 0
    skipped: dict[str, SkippedEvent] = field(default_factory=dict)

    def next_player_id(self) -> str:
        self.players += 1
        return f"player-{self.players}"

    def note_skipped(self, items: tuple[SkippedEvent, ...] | list[SkippedEvent]) -> None:
        """Record events a projection on this page could not use."""
        for item in items:
            self.skipped.setdefault(item.event_id, item)


class Console:
    """The console's handlers, bound to one hub (log, artifact store, config)."""

    def __init__(
        self,
        log: EventLog,
        store: ArtifactStore,
        config: ConsoleConfig,
        *,
        clock: Clock = utcnow,
    ) -> None:
        self.log = log
        self.store = store
        self.config = config
        self.clock = clock
        self.csrf = CsrfSigner(
            config.csrf_secret,
            max_age=config.csrf_max_age,
            now=lambda: clock().timestamp(),
        )

    # -- identity and request checks ---------------------------------------

    def identity(self, request: Request) -> Actor | None:
        """Return the human making this request, or ``None`` (read-only)."""
        if self.config.trust_tailscale_headers:
            login = request.headers.get("tailscale-user-login")
            peer = request.client.host if request.client else None
            if login is not None and _is_loopback(peer):
                # Only tailscale serve (on loopback) may vouch for a login. A malformed
                # one refuses writes rather than silently becoming the operator.
                return Actor("human", login) if _NAME_RE.fullmatch(login) else None
        if self.config.operator:
            return Actor("human", self.config.operator)
        return None

    @staticmethod
    def check_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        source = origin if origin is not None else request.headers.get("referer")
        if source is None:
            raise ConsoleError(
                403,
                "origin_missing",
                "this write carried neither an Origin nor a Referer header",
                "Submit the form from the console page itself.",
            )
        parts = urlsplit(source) if source != "null" else None
        host = request.headers.get("host", "")
        if parts is None or parts.scheme not in ("http", "https") or not host:
            raise ConsoleError(403, "origin_mismatch", "this write came from another origin")
        if parts.netloc.lower() != host.lower():
            raise ConsoleError(403, "origin_mismatch", "this write came from another origin")

    def require_actor(self, request: Request) -> Actor:
        actor = self.identity(request)
        if actor is None:
            raise ConsoleError(
                403,
                "no_identity",
                "the console has no human identity for this request, so it is read-only",
                "Start it with --operator NAME, or with --trust-tailscale-headers behind "
                "tailscale serve.",
            )
        return actor

    async def write_form(self, request: Request) -> tuple[Actor, FormData, str]:
        """Run every write check and return the actor, the form and its idempotency key."""
        self.check_origin(request)
        actor = self.require_actor(request)
        form = await read_form(request)
        token = form.get("csrf")
        if not isinstance(token, str) or not self.csrf.verify(token, _ident(actor)):
            raise ConsoleError(
                403,
                "csrf_failed",
                "the form's CSRF token is missing, expired or not yours",
                "Reload the page and try again.",
            )
        idem = form.get("idem")
        if not isinstance(idem, str) or not _IDEM_RE.fullmatch(idem):
            raise ConsoleError(400, "idempotency_key_missing", "the form has no valid idem key")
        return actor, form, f"console:{idem}"

    # -- rendering -----------------------------------------------------------

    def page_context(self, request: Request) -> _Page:
        actor = self.identity(request)
        csrf = self.csrf.issue(_ident(actor)) if actor is not None else None
        return _Page(request, actor, csrf, secrets.token_urlsafe(16))

    def hidden(self, ctx: _Page) -> Html:
        """The CSRF token and a fresh idempotency key for one form."""
        return fmt(
            '<input type="hidden" name="csrf" value="{csrf}">'
            '<input type="hidden" name="idem" value="{idem}">',
            csrf=ctx.csrf or "",
            idem=secrets.token_hex(16),
        )

    def render(
        self, ctx: _Page, title: str, nav: str, body: Html, *, status: int = 200
    ) -> HTMLResponse:
        banner = Html("")
        try:
            state = control.status(self.log)
        except Exception:
            ref = _log_failure("reading the emergency-stop state")
            banner = fmt(
                '<div class="banner" role="alert">Could not read the emergency-stop state '
                '(reference {ref}). Check <a href="/control">Control</a>.</div>',
                ref=ref,
            )
            state = control.ControlStatus(())
        ctx.note_skipped(state.skipped)
        stops = state.stops
        if stops:
            first = stops[0]
            more = f" (+{len(stops) - 1} more)" if len(stops) > 1 else ""
            banner = fmt(
                '<div class="banner" role="alert">STOPPED: {scope}{more} by {who} at {at}: '
                '{reason} <a href="/control">Control</a></div>',
                scope=str(first.scope),
                more=more,
                who=first.by.id,
                at=_when(first.at),
                reason=first.reason,
            )
        links = join(
            fmt(
                '<a href="{href}" class="{cls}"{current}>{label}</a>',
                href=href,
                cls="stop" if key == "control" else "",
                current=Html(' aria-current="page"' if key == nav else ""),
                label=label,
            )
            for key, href, label in (
                ("reviews", "/", "Reviews"),
                ("leases", "/leases", "Leases"),
                ("approvals", "/approvals", "Approvals"),
                ("control", "/control", "STOP"),
            )
        )
        who = (
            fmt("Signed in as <strong>{id}</strong> (human)", id=ctx.actor.id)
            if ctx.actor
            else Html("Read-only: no operator identity")
        )
        flash_code = ctx.request.query_params.get("done")
        flash = (
            fmt('<p class="flash" role="status">{msg}</p>', msg=_FLASH[flash_code])
            if flash_code in _FLASH
            else Html("")
        )
        document = fmt(
            "<!doctype html><html lang=en><head><meta charset=utf-8>"
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="color-scheme" content="light dark">'
            "<title>{title} - aar console</title>"
            '<style nonce="{nonce}">{css}</style></head><body>'
            '<header><span class="brand">aar console</span><nav>{links}</nav>'
            '<span class="who">{who}</span></header>{banner}'
            "<main><h1>{title}</h1>{flash}{skipped}{body}</main>"
            '<script nonce="{nonce}">{script}</script></body></html>',
            title=title,
            nonce=ctx.nonce,
            css=Html(_CSS),
            links=links,
            who=who,
            banner=banner,
            flash=flash,
            skipped=_skipped_notice(ctx.skipped.values()),
            body=body,
            script=Html(_SCRIPT),
        )
        csp = (
            f"default-src 'none'; style-src 'nonce-{ctx.nonce}'; "
            f"script-src 'nonce-{ctx.nonce}'; img-src 'self'; media-src 'self'; "
            "form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        # nosniff, X-Frame-Options, Referrer-Policy and no-store come from
        # SecurityHeaders, which covers every response; a page adds its nonce CSP.
        return HTMLResponse(document, status_code=status, headers={"Content-Security-Policy": csp})

    def error_response(self, request: Request, error: ConsoleError) -> Response:
        if _wants_json(request):
            body = {"error": {"code": error.code, "message": error.message, "hint": error.hint}}
            return JSONResponse(body, status_code=error.status)
        ctx = self.page_context(request)
        hint = fmt("<p>{hint}</p>", hint=error.hint) if error.hint else Html("")
        body = fmt(
            '<section class="error" role="alert"><p><strong>{status}</strong> '
            "<code>{code}</code></p><p>{message}</p>{hint}"
            '<p><a href="{back}">Back</a></p></section>',
            status=error.status,
            code=error.code,
            message=error.message,
            hint=hint,
            back=_back_path(request),
        )
        return self.render(ctx, "Something went wrong", "", body, status=error.status)

    def write_disabled_note(self, ctx: _Page) -> Html:
        if ctx.actor is not None:
            return Html("")
        return Html(
            '<p class="meta">Read-only: start the console with --operator NAME (or behind '
            "tailscale serve with --trust-tailscale-headers) to act.</p>"
        )

    # -- reviews -------------------------------------------------------------

    def project_names(self, ctx: _Page) -> dict[str, str]:
        registry = Registry(self.log)
        ctx.note_skipped(registry.skipped())
        return {project.id: project.name for project in registry.list()}

    def media(self, ctx: _Page, artifact_id: str, label: str) -> tuple[Html, str | None]:
        """Return a player (or link) for one render, and the player's element id."""
        meta = self.store.get_meta(artifact_id)
        mime = meta.mime if meta else None
        href = url("artifacts", artifact_id)
        if mime and mime.startswith("video/") and mime in _INLINE_MIME:
            player = ctx.next_player_id()
            return (
                fmt(
                    '<video id="{pid}" controls playsinline preload="metadata" src="{src}#t=0.1" '
                    'aria-label="{label}"></video>',
                    pid=player,
                    src=href,
                    label=label,
                ),
                player,
            )
        if mime and mime.startswith("image/") and mime in _INLINE_MIME:
            return fmt(
                '<img class="render" src="{src}" alt="{label}">', src=href, label=label
            ), None
        return fmt('<p><a href="{src}">Download {label}</a></p>', src=href, label=label), None

    def review_card(
        self, ctx: _Page, review: reviews.Review, names: dict[str, str], *, detail: bool
    ) -> Html:
        detail_href = url("reviews", review.project_id, review.id)
        title = (
            fmt("<h2>{title}</h2>", title=review.title)
            if detail
            else fmt('<h2><a href="{href}">{title}</a></h2>', href=detail_href, title=review.title)
        )
        project = names.get(review.project_id, review.project_id)
        meta = fmt(
            '<p class="meta">{status} {project} &middot; requested by {who} &middot; {when}'
            "{timeline}{fps}</p>",
            status=_badge(review.status),
            project=project,
            who=review.requested_by or "unknown",
            when=_when(review.created),
            timeline=f" \u00b7 timeline {review.timeline}" if review.timeline else "",
            fps=f" \u00b7 {review.fps:g} fps" if review.fps else "",
        )
        players: list[Html] = []
        first_player: str | None = None
        shown = review.renders if detail else review.renders[:1]
        for index, artifact_id in enumerate(shown, start=1):
            block, player = self.media(ctx, artifact_id, f"render {index} of {review.title}")
            players.append(block)
            first_player = first_player or player
        if not review.renders:
            players.append(Html('<p class="empty">No render attached.</p>'))
        extra = ""
        if not detail and len(review.renders) > 1:
            extra = fmt(
                '<p class="renders">+{n} more render(s) on the <a href="{href}">detail page</a>'
                "</p>",
                n=len(review.renders) - 1,
                href=detail_href,
            )
        notes = self.notes_list(review, first_player)
        decision = self.decision_block(ctx, review, return_to="detail" if detail else "queue")
        open_for_notes = detail and review.pending and review.decision is None
        note_form = self.note_form(ctx, review, first_player) if open_for_notes else Html("")
        return fmt(
            '<article class="card">{title}{meta}<div class="review-body"><div class="media">'
            '{players}{extra}</div><div class="side"><h3>Notes</h3>{notes}{decision}</div>'
            "{note_form}</div></article>",
            title=title,
            meta=meta,
            players=join(players),
            extra=extra,
            notes=notes,
            note_form=note_form,
            decision=decision,
        )

    @staticmethod
    def notes_list(review: reviews.Review, player: str | None) -> Html:
        if not review.notes:
            return Html('<p class="empty">No notes yet.</p>')
        items = []
        for note in review.notes:
            if player and review.fps:
                stamp = fmt(
                    '<button type="button" class="tc" data-player="{pid}" data-seek="{sec}" '
                    'title="Seek the player to frame {frame}">{tc}</button>',
                    pid=player,
                    sec=f"{note.frame / review.fps:.3f}",
                    frame=note.frame,
                    tc=note.timecode,
                )
            else:
                stamp = fmt(
                    '<button type="button" class="tc" disabled title="frame {frame}">{tc}</button>',
                    frame=note.frame,
                    tc=note.timecode,
                )
            items.append(
                fmt(
                    '<li>{stamp}<div class="note-text">{text}'
                    '<div class="meta">{kind} {who} &middot; frame {frame}</div></div></li>',
                    stamp=stamp,
                    text=note.text,
                    kind=_badge(note.author.kind),
                    who=note.author.id,
                    frame=note.frame,
                )
            )
        return fmt('<ol class="notes">{items}</ol>', items=join(items))

    def decision_block(self, ctx: _Page, review: reviews.Review, *, return_to: str) -> Html:
        if review.decision is not None:
            d = review.decision
            return fmt(
                '<div class="decision {kind}"><p>{badge} by <strong>{who}</strong> at {at}</p>'
                '<p class="note-text">{reason}</p></div>',
                kind=d.decision,
                badge=_badge(d.decision),
                who=d.by.id,
                at=_when(d.ts),
                reason=d.reason,
            )
        if review.withdrawal is not None:
            w = review.withdrawal
            return fmt(
                '<div class="decision"><p>{badge} by <strong>{who}</strong> at {at}</p>'
                '<p class="note-text">{reason}</p></div>',
                badge=_badge("withdrawn", "cancelled"),
                who=f"{w.by.kind}:{w.by.id}",
                at=_when(w.ts),
                reason=w.reason,
            )
        if not review.pending:
            return fmt(
                '<p class="meta">Closed as {status} without a recorded human decision.</p>',
                status=review.status,
            )
        if ctx.actor is None:
            return self.write_disabled_note(ctx)
        reason_id = f"reason-{ctx.next_player_id()}"
        return fmt(
            '<form method="post" action="{action}">{hidden}'
            '<input type="hidden" name="return_to" value="{ret}">'
            '<label for="{rid}">Reason (required, the agent reads this)</label>'
            '<textarea id="{rid}" name="reason" required maxlength="{max}"></textarea>'
            '<div class="actions"><button class="approve" name="decision" value="approve">'
            "Approve</button>"
            '<button class="reject" name="decision" value="reject">Reject</button></div></form>',
            action=url("reviews", review.project_id, review.id, "decide"),
            hidden=self.hidden(ctx),
            ret=return_to,
            rid=reason_id,
            max=reviews.MAX_REASON,
        )

    def note_form(self, ctx: _Page, review: reviews.Review, player: str | None) -> Html:
        if ctx.actor is None:
            return Html("")
        frame_id = f"frame-{ctx.next_player_id()}"
        grab = Html("")
        if player and review.fps:
            grab = fmt(
                '<button type="button" class="plain" data-frame-from="{pid}" data-input="{fid}" '
                'data-fps="{fps}">Use player position</button>',
                pid=player,
                fid=frame_id,
                fps=f"{review.fps:g}",
            )
        timecode = Html("")
        if review.fps is None:
            timecode = Html(
                '<div><label for="note-tc">Timecode (HH:MM:SS:FF)</label>'
                '<input id="note-tc" name="timecode" required pattern="[0-9]{2}:[0-9]{2}:'
                '[0-9]{2}[:;][0-9]{2}" placeholder="01:00:00:00"></div>'
            )
        return fmt(
            '<form method="post" action="{action}" class="subform">{hidden}'
            "<h3>Add a note at a frame</h3>"
            '<div class="row"><div><label for="{fid}">Frame (0-based, in the render)</label>'
            '<input id="{fid}" name="frame" inputmode="numeric" pattern="[0-9]+" required></div>'
            "{timecode}</div>"
            '<label for="note-text">Note</label>'
            '<textarea id="note-text" name="text" required maxlength="{max}"></textarea>'
            '<div class="actions"><button class="plain" type="submit">Add note</button>{grab}'
            "</div></form>",
            action=url("reviews", review.project_id, review.id, "notes"),
            hidden=self.hidden(ctx),
            fid=frame_id,
            timecode=timecode,
            max=reviews.MAX_NOTE,
            grab=grab,
        )

    def queue_body(self, ctx: _Page) -> Html:
        names = self.project_names(ctx)
        scan = reviews.scan_reviews(self.log)
        ctx.note_skipped(scan.skipped)
        everything = list(scan.reviews)
        pending = [r for r in everything if r.pending]
        decided = sorted(
            (r for r in everything if not r.pending),
            key=lambda r: (r.updated or "", r.id),
            reverse=True,
        )[:10]
        cards = (
            join(self.review_card(ctx, r, names, detail=False) for r in pending)
            if pending
            else Html('<p class="empty card">Nothing is waiting for review.</p>')
        )
        recent = Html("")
        if decided:
            rows = join(
                fmt(
                    '<tr><td data-label="Review"><a href="{href}">{title}</a></td>'
                    '<td data-label="Project">{project}</td><td data-label="Outcome">{badge}</td>'
                    '<td data-label="By">{who}</td><td data-label="When">{when}</td></tr>',
                    href=url("reviews", r.project_id, r.id),
                    title=r.title,
                    project=names.get(r.project_id, r.project_id),
                    badge=_badge(r.decision.decision if r.decision else r.status),
                    who=r.decision.by.id if r.decision else "-",
                    when=_when(r.updated),
                )
                for r in decided
            )
            recent = fmt(
                '<h2>Recently closed</h2><table class="board"><thead><tr><th>Review</th>'
                "<th>Project</th><th>Outcome</th><th>By</th><th>When</th></tr></thead>"
                "<tbody>{rows}</tbody></table>",
                rows=rows,
            )
        return fmt(
            '<p class="meta">{n} waiting for a human.</p>{cards}{recent}',
            n=len(pending),
            cards=cards,
            recent=recent,
        )

    def detail_body(self, ctx: _Page, project_id: str, review_id: str) -> Html:
        review, skipped = reviews.review_view(self.log, project_id, review_id)
        ctx.note_skipped(skipped)
        if review is None:
            raise ConsoleError(
                404, "review_not_found", f"no review {review_id!r} in project {project_id!r}"
            )
        subject = fmt(
            '<p class="meta">Under review: node <code>{subject}</code> &middot; review '
            "<code>{id}</code> &middot; project <code>{pid}</code></p>",
            subject=review.subject or "unknown",
            id=review.id,
            pid=review.project_id,
        )
        return join([subject, self.review_card(ctx, review, self.project_names(ctx), detail=True)])

    # -- leases ----------------------------------------------------------------

    def leases_body(self, ctx: _Page) -> Html:
        board = lease_board(self.log, clock=self.clock)
        ctx.note_skipped(board.skipped)
        if board.leases:
            rows = join(
                fmt(
                    '<tr><td data-label="Scope">{scope}</td><td data-label="Key">'
                    '<span class="mono">{key}</span></td><td data-label="Holder">{kind} {holder}'
                    '</td><td data-label="Node">{node}</td><td data-label="Since">{since}</td>'
                    '<td data-label="Expires">{expires}</td></tr>',
                    scope=lease.scope,
                    key=" / ".join(lease.key),
                    kind=_badge(lease.holder_kind),
                    holder=lease.holder_id,
                    node=lease.node_id,
                    since=_when(lease.since),
                    expires=_when(lease.expires_at) if lease.expires_at else "held until released",
                )
                for lease in board.leases
            )
            leases = fmt(
                '<table class="board"><thead><tr><th>Scope</th><th>Key</th><th>Holder</th>'
                "<th>Node</th><th>Since</th><th>Expires</th></tr></thead><tbody>{rows}</tbody>"
                "</table>",
                rows=rows,
            )
        else:
            leases = Html('<p class="empty">No leases are held.</p>')
        if board.presence:
            people = join(
                fmt(
                    "<li>{badge} <strong>{who}</strong> active on node <code>{node}</code> "
                    "since {since}</li>",
                    badge=_badge("human"),
                    who=entry.human_id or "a human",
                    node=entry.node_id,
                    since=_when(entry.since),
                )
                for entry in board.presence
            )
            presence = fmt("<ul>{people}</ul>", people=people)
        else:
            presence = Html('<p class="empty">No human activity reported.</p>')
        return fmt(
            '<p class="meta">As of {as_of}. Expired leases, and leases a node no longer holds '
            "by its latest snapshot, are left out.</p>{stale}"
            '<section class="card"><h2>Who holds what</h2>{leases}</section>'
            '<section class="card"><h2>Humans at work</h2>{presence}</section>',
            as_of=_when(board.as_of),
            stale=self.stale_card(board),
            leases=leases,
            presence=presence,
        )

    @staticmethod
    def stale_card(board: LeaseBoard) -> Html:
        """Which nodes' leases may be out of date on this page, and why; empty if none."""
        if not board.stale:
            return Html("")
        gaps = (
            fmt(
                "<li>Node <code>{node}</code> dropped {n} feed event(s) at {at} and has not "
                "reconciled since.</li>",
                node=gap.node_id,
                n=gap.dropped,
                at=_when(gap.at),
            )
            for gap in board.gaps
        )
        refused = (
            fmt(
                "<li>Node <code>{node}</code>: a second feed process (<code>{nonce}</code>) "
                "was refused at {at}; the node's current one is <code>{owner}</code>. Two "
                "processes must not share a node id.</li>",
                node=entry.node_id,
                nonce=entry.nonce,
                at=_when(entry.at),
                owner=entry.owner,
            )
            for entry in board.superseded
        )
        return fmt(
            '<section class="card" role="status"><h2>This view may be stale</h2>'
            "<ul>{items}</ul></section>",
            items=join([*gaps, *refused]),
        )

    # -- approvals ---------------------------------------------------------------

    def approvals_body(self, ctx: _Page) -> Html:
        now = self.clock()
        view = approvals.approvals_view(self.log)
        ctx.note_skipped(view.skipped)
        tokens = list(view.tokens)
        active = [t for t in tokens if t.state(now) == "active"]
        inactive = [t for t in tokens if t.state(now) != "active"][:50]
        grant = self.write_disabled_note(ctx)
        if ctx.actor is not None:
            scopes = join(
                fmt('<option value="{v}">{v}</option>', v=s.value) for s in approvals.ApprovalScope
            )
            ttls = join(
                fmt(
                    '<option value="{v}"{sel}>{label}</option>',
                    v=v,
                    label=label,
                    sel=Html(" selected" if v == 3600 else ""),
                )
                for v, label in _TTL_CHOICES
            )
            grant = fmt(
                '<form method="post" action="/approvals/grant" class="card">{hidden}'
                "<h2>Grant a single-use approval</h2>"
                '<div class="row"><div><label for="g-scope">Scope</label>'
                '<select id="g-scope" name="scope">{scopes}</select></div>'
                '<div><label for="g-ttl">Expires after</label>'
                '<select id="g-ttl" name="ttl">{ttls}</select></div></div>'
                '<label for="g-target">Target (for example a project id or database name)</label>'
                '<input id="g-target" name="target" required maxlength="{tmax}">'
                '<label for="g-reason">Reason</label>'
                '<textarea id="g-reason" name="reason" required maxlength="{rmax}"></textarea>'
                '<div class="actions"><button class="approve" type="submit">Grant token</button>'
                "</div></form>",
                hidden=self.hidden(ctx),
                scopes=scopes,
                ttls=ttls,
                tmax=approvals.MAX_TARGET,
                rmax=approvals.MAX_REASON,
            )

        def row(token: approvals.Approval, *, revocable: bool) -> Html:
            action = Html("")
            if revocable and ctx.actor is not None:
                action = fmt(
                    '<form method="post" action="/approvals/revoke">{hidden}'
                    '<input type="hidden" name="token_id" value="{tid}">'
                    '<input name="reason" required placeholder="Why revoke?" maxlength="{rmax}" '
                    'aria-label="Reason to revoke {tid}">'
                    '<div class="actions"><button class="danger" type="submit">Revoke</button>'
                    "</div></form>",
                    hidden=self.hidden(ctx),
                    tid=token.token_id,
                    rmax=approvals.MAX_REASON,
                )
            state = token.state(now)
            return fmt(
                '<tr><td data-label="Token"><span class="mono">{tid}</span></td>'
                '<td data-label="Scope"><code>{scope}</code></td>'
                '<td data-label="Target"><span class="mono">{target}</span></td>'
                '<td data-label="State">{state}</td><td data-label="Reason">{reason}</td>'
                '<td data-label="Granted">{by} at {at}</td><td data-label="Expires">{exp}</td>'
                '<td data-label="Action">{action}</td></tr>',
                tid=token.token_id,
                scope=token.scope.value,
                target=token.target,
                state=_badge(state),
                reason=token.reason,
                by=token.granted_by.id,
                at=_when(token.granted_at, short=True),
                exp=_when(token.expires_at, short=True),
                action=action,
            )

        head = Html(
            '<table class="board"><thead><tr><th>Token</th><th>Scope</th><th>Target</th>'
            "<th>State</th><th>Reason</th><th>Granted</th><th>Expires</th><th></th></tr>"
            "</thead><tbody>"
        )
        active_html = (
            fmt(
                "{head}{rows}</tbody></table>",
                head=head,
                rows=join(row(t, revocable=True) for t in active),
            )
            if active
            else Html('<p class="empty">No active tokens.</p>')
        )
        inactive_html = (
            fmt(
                "{head}{rows}</tbody></table>",
                head=head,
                rows=join(row(t, revocable=False) for t in inactive),
            )
            if inactive
            else Html('<p class="empty">None yet.</p>')
        )
        return fmt(
            '<p class="meta">Agents can list these but never grant them. Each token is good '
            "for one use of one scope on one target.</p>{grant}"
            '<section class="card"><h2>Active</h2>{active}</section>'
            '<section class="card"><h2>Used, revoked or expired</h2>{inactive}</section>',
            grant=grant,
            active=active_html,
            inactive=inactive_html,
        )

    # -- control -------------------------------------------------------------------

    def panel(self, ctx: _Page, what: str, build: Callable[[_Page], Html]) -> Html:
        """Render one panel; if it fails, show that (with a log reference) instead.

        Used on the STOP page so one broken projection can never hide the stop and
        resume controls.
        """
        try:
            return build(ctx)
        except Exception:
            ref = _log_failure(f"rendering the {what} panel")
            return fmt(
                '<section class="error" role="alert"><p>The {what} panel failed to load '
                "(reference {ref}). The other controls on this page still work.</p></section>",
                what=what,
                ref=ref,
            )

    def control_body(self, ctx: _Page) -> Html:
        current = self.panel(ctx, "current stops", self.stops_panel)
        if ctx.actor is None:
            return join([current, self.write_disabled_note(ctx)])
        big = fmt(
            '<form method="post" action="/control/stop" class="card">{hidden}'
            '<input type="hidden" name="scope" value="all">'
            '<button class="big-stop" type="submit">STOP ALL</button>'
            '<label for="stop-reason">Reason (optional)</label>'
            '<input id="stop-reason" name="reason" maxlength="{rmax}" '
            'placeholder="Emergency stop from the console"></form>',
            hidden=self.hidden(ctx),
            rmax=control.MAX_REASON,
        )
        scoped = self.panel(ctx, "stop one project or node", self.targets_panel)
        return join([current, big, scoped])

    def safe_project_names(self, ctx: _Page) -> dict[str, str]:
        """Project names for labels; empty (and logged) if the registry cannot be read."""
        try:
            return self.project_names(ctx)
        except Exception:
            _log_failure("reading project names for the STOP page")
            return {}

    def stops_panel(self, ctx: _Page) -> Html:
        state = control.status(self.log)
        ctx.note_skipped(state.skipped)
        if not state.stops:
            return Html('<p class="card">Everything is running. No stop is in effect.</p>')
        names = self.safe_project_names(ctx)
        items = []
        for stop in state.stops:
            label = (
                names.get(stop.scope.id or "", stop.scope.id or "")
                if stop.scope.kind == "project"
                else str(stop.scope)
            )
            resume = self.write_disabled_note(ctx)
            if ctx.actor is not None:
                resume = fmt(
                    '<form method="post" action="/control/resume">{hidden}'
                    '<input type="hidden" name="scope" value="{scope}">'
                    '<label for="r-{n}">Why is it safe to resume?</label>'
                    '<input id="r-{n}" name="reason" required maxlength="{rmax}">'
                    '<div class="actions"><button class="approve" type="submit">Resume '
                    "{label}</button></div></form>",
                    hidden=self.hidden(ctx),
                    scope=str(stop.scope),
                    n=len(items),
                    rmax=control.MAX_REASON,
                    label=label,
                )
            items.append(
                fmt(
                    '<section class="card stopped"><h2>STOPPED: {label}</h2>'
                    '<p class="note-text">{reason}</p>'
                    '<p class="meta">by {kind} <strong>{who}</strong> at {at} &middot; scope '
                    "<code>{scope}</code></p>{resume}</section>",
                    label=label,
                    reason=stop.reason,
                    kind=_badge(stop.by.kind),
                    who=stop.by.id,
                    at=_when(stop.at),
                    scope=str(stop.scope),
                    resume=resume,
                )
            )
        return join(items)

    def targets_panel(self, ctx: _Page) -> Html:
        names = self.project_names(ctx)
        targets: list[tuple[str, str]] = [(f"project:{pid}", name) for pid, name in names.items()]
        board = lease_board(self.log, clock=self.clock)
        ctx.note_skipped(board.skipped)
        nodes = sorted(
            {lease.node_id for lease in board.leases} | {p.node_id for p in board.presence}
        )
        targets += [(f"node:{node}", f"node {node}") for node in nodes]
        if not targets:
            return Html("")
        buttons = join(
            fmt(
                '<form method="post" action="/control/stop">{hidden}'
                '<input type="hidden" name="scope" value="{scope}">'
                '<button class="danger" type="submit">Stop {label}</button></form>',
                hidden=self.hidden(ctx),
                scope=scope,
                label=label,
            )
            for scope, label in targets
        )
        return fmt(
            '<section class="card"><h2>Stop one project or node</h2>'
            '<div class="actions">{buttons}</div></section>',
            buttons=buttons,
        )


def _log_failure(what: str) -> str:
    """Log the current exception with a short reference a human can quote; return it."""
    ref = secrets.token_hex(4)
    logger.exception("console failure [%s] while %s", ref, what)
    return ref


def _skipped_notice(items: Iterable[SkippedEvent]) -> Html:
    """'N malformed events ignored' with their ids, or nothing."""
    listed = sorted(items, key=lambda item: item.seq)
    if not listed:
        return Html("")
    rows = join(
        fmt(
            "<li><code>{id}</code> {type}: {reason}</li>",
            id=item.event_id,
            type=item.type,
            reason=item.reason,
        )
        for item in listed[:MAX_SKIPPED_LISTED]
    )
    more = (
        fmt("<li>and {n} more</li>", n=len(listed) - MAX_SKIPPED_LISTED)
        if len(listed) > MAX_SKIPPED_LISTED
        else Html("")
    )
    return fmt(
        '<details class="skipped"><summary>{n} malformed event{s} ignored</summary>'
        "<p>The log is append-only, so these stay, but nothing on this page uses them.</p>"
        "<ul>{rows}{more}</ul></details>",
        n=len(listed),
        s="" if len(listed) == 1 else "s",
        rows=rows,
        more=more,
    )


def _back_path(request: Request) -> str:
    """The same-origin page a failed form came from, or the queue."""
    referer = request.headers.get("referer")
    if not referer:
        return "/"
    parts = urlsplit(referer)
    if parts.netloc.lower() != request.headers.get("host", "").lower():
        return "/"
    path = parts.path
    if not path.startswith("/") or path.startswith("//"):
        return "/"
    return quote(path, safe="/%-._~")


def _ident(actor: Actor | None) -> str:
    return "" if actor is None else f"{actor.kind}:{actor.id}"


def _field(form: FormData, name: str, *, required: bool = True) -> str:
    value = form.get(name)
    if value is None:
        if required:
            raise ConsoleError(422, "invalid_input", f"{name}: is required")
        return ""
    if not isinstance(value, str):
        raise ConsoleError(422, "invalid_input", f"{name}: must be text, not a file")
    return value


def _int_field(form: FormData, name: str) -> int:
    raw = _field(form, name).strip()
    if not _UINT_RE.fullmatch(raw):
        raise ConsoleError(422, "invalid_input", f"{name}: must be a whole number (0-9 digits)")
    return int(raw)


# -- artifacts --------------------------------------------------------------------


class _Unsatisfiable(Exception):
    pass


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Return the inclusive byte range a ``Range`` header asks for, or ``None`` for all.

    Only a single ``bytes=`` range is honoured; anything else is ignored (RFC 9110
    lets a server serve the whole representation instead). Raises ``_Unsatisfiable``
    for a well-formed range that lies outside the content.
    """
    if header is None:
        return None
    match = _RANGE_RE.fullmatch(header.strip())
    if match is None:
        return None
    first, last = match.groups()
    if not first and not last:
        return None
    if not first:
        suffix = int(last)
        if suffix == 0 or size == 0:
            raise _Unsatisfiable
        return max(0, size - suffix), size - 1
    start = int(first)
    if start >= size:
        raise _Unsatisfiable
    end = size - 1 if not last else min(int(last), size - 1)
    if end < start:
        return None
    return start, end


def _stream(handle: BinaryIO, start: int, length: int) -> Iterator[bytes]:
    try:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(_CHUNK, remaining))
            if not chunk:
                return
            remaining -= len(chunk)
            yield chunk
    finally:
        handle.close()


# -- wiring ----------------------------------------------------------------------------


def build_app(
    log: EventLog,
    store: ArtifactStore,
    config: ConsoleConfig | None = None,
    *,
    clock: Clock = utcnow,
) -> ASGIApp:
    """Build the console ASGI app over one hub."""
    config = config or ConsoleConfig()
    console = Console(log, store, config, clock=clock)

    async def page(
        request: Request, title: str, nav: str, body: Callable[[_Page], Html]
    ) -> Response:
        ctx = console.page_context(request)
        try:
            content = await run_in_threadpool(body, ctx)
        except ConsoleError as exc:
            return console.error_response(request, exc)
        return console.render(ctx, title, nav, content)

    async def write(request: Request, action: Callable[[Actor, FormData, str], str]) -> Response:
        try:
            actor, form, key = await console.write_form(request)
            location = await run_in_threadpool(action, actor, form, key)
        except Exception as exc:
            mapped = _map_error(exc)
            if mapped is None:
                raise
            logger.info("console write refused: %s %s: %s", request.url.path, mapped.code, exc)
            return console.error_response(request, mapped)
        logger.info("console write by %s: %s", _ident(actor), request.url.path)
        return RedirectResponse(location, status_code=303)

    async def queue(request: Request) -> Response:
        return await page(request, "Review queue", "reviews", console.queue_body)

    async def review_detail(request: Request) -> Response:
        pid = request.path_params["project_id"]
        rid = request.path_params["review_id"]
        return await page(
            request, "Review", "reviews", lambda ctx: console.detail_body(ctx, pid, rid)
        )

    async def review_decide(request: Request) -> Response:
        pid = request.path_params["project_id"]
        rid = request.path_params["review_id"]

        def action(actor: Actor, form: FormData, key: str) -> str:
            decision = _field(form, "decision")
            reviews.decide(
                log, pid, rid, decision, _field(form, "reason"), actor, idempotency_key=key
            )
            done = "approved" if decision == "approve" else "rejected"
            if _field(form, "return_to", required=False) == "detail":
                return url("reviews", pid, rid, done=done)
            return url("", done=done)

        return await write(request, action)

    async def review_note(request: Request) -> Response:
        pid = request.path_params["project_id"]
        rid = request.path_params["review_id"]

        def action(actor: Actor, form: FormData, key: str) -> str:
            timecode = _field(form, "timecode", required=False).strip() or None
            reviews.add_note(
                log,
                pid,
                rid,
                _int_field(form, "frame"),
                _field(form, "text"),
                actor,
                timecode=timecode,
                idempotency_key=key,
            )
            return url("reviews", pid, rid, done="noted")

        return await write(request, action)

    async def artifact(request: Request) -> Response:
        artifact_id = request.path_params["artifact_id"]
        if not _ARTIFACT_RE.fullmatch(artifact_id):
            return console.error_response(
                request,
                ConsoleError(
                    400, "invalid_artifact_id", "artifact ids are 64 lowercase hex characters"
                ),
            )
        try:
            handle = await run_in_threadpool(store.open, artifact_id)
        except FileNotFoundError:
            return console.error_response(
                request, ConsoleError(404, "artifact_not_found", f"no artifact {artifact_id}")
            )
        except CorruptArtifact as exc:
            logger.error("corrupt artifact %s: %s", artifact_id, exc)
            return console.error_response(request, ConsoleError(500, "artifact_corrupt", str(exc)))
        try:
            size = os.fstat(handle.fileno()).st_size
            meta = store.get_meta(artifact_id)
            mime = meta.mime if meta and meta.mime in _INLINE_MIME else None
            headers = {
                "Accept-Ranges": "bytes",
                "ETag": f'"{artifact_id}"',
                "Cache-Control": "private, max-age=31536000, immutable",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": (
                    "sandbox; default-src 'none'; media-src 'self'; img-src 'self'; "
                    "style-src 'unsafe-inline'"
                ),
            }
            if mime is None:
                headers["Content-Disposition"] = f'attachment; filename="{artifact_id}"'
            try:
                wanted = parse_range(request.headers.get("range"), size)
            except _Unsatisfiable:
                handle.close()
                headers["Content-Range"] = f"bytes */{size}"
                return Response(b"", status_code=416, headers=headers)
            start, end = wanted if wanted is not None else (0, size - 1)
            length = max(0, end - start + 1)
            headers["Content-Length"] = str(length)
            status = 200
            if wanted is not None:
                status = 206
                headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            media_type = mime or "application/octet-stream"
            if request.method == "HEAD":
                handle.close()
                return Response(b"", status_code=status, headers=headers, media_type=media_type)
        except BaseException:
            handle.close()
            raise
        return StreamingResponse(
            _stream(handle, start, length),
            status_code=status,
            headers=headers,
            media_type=media_type,
        )

    async def leases(request: Request) -> Response:
        return await page(request, "Lease board", "leases", console.leases_body)

    async def approvals_page(request: Request) -> Response:
        return await page(request, "Approvals", "approvals", console.approvals_body)

    async def approvals_grant(request: Request) -> Response:
        def action(actor: Actor, form: FormData, key: str) -> str:
            approvals.grant(
                log,
                _field(form, "scope"),
                _field(form, "target"),
                _int_field(form, "ttl"),
                _field(form, "reason"),
                actor,
                idempotency_key=key,
                clock=clock,
            )
            return url("approvals", done="granted")

        return await write(request, action)

    async def approvals_revoke(request: Request) -> Response:
        def action(actor: Actor, form: FormData, key: str) -> str:
            approvals.revoke(
                log,
                _field(form, "token_id"),
                _field(form, "reason"),
                actor,
                idempotency_key=key,
                clock=clock,
            )
            return url("approvals", done="revoked")

        return await write(request, action)

    async def control_page(request: Request) -> Response:
        return await page(request, "Emergency stop", "control", console.control_body)

    async def control_stop(request: Request) -> Response:
        def action(actor: Actor, form: FormData, key: str) -> str:
            reason = _field(form, "reason", required=False).strip()
            control.stop(
                log,
                _field(form, "scope"),
                reason or "Emergency stop from the console",
                actor,
                idempotency_key=key,
            )
            return url("control", done="stopped")

        return await write(request, action)

    async def control_resume(request: Request) -> Response:
        def action(actor: Actor, form: FormData, key: str) -> str:
            control.resume(
                log, _field(form, "scope"), _field(form, "reason"), actor, idempotency_key=key
            )
            return url("control", done="resumed")

        return await write(request, action)

    async def http_error(request: Request, exc: Exception) -> Response:
        if not isinstance(exc, HTTPException):  # pragma: no cover - registered for it only
            raise exc
        codes = {404: "not_found", 405: "method_not_allowed", 413: "body_too_large"}
        error = ConsoleError(
            exc.status_code,
            codes.get(exc.status_code, "http_error"),
            str(exc.detail),
            "Writes are POST-only forms on the console's pages."
            if exc.status_code == 405
            else None,
        )
        response = console.error_response(request, error)
        for name, value in (exc.headers or {}).items():
            response.headers[name] = value
        return response

    async def server_error(request: Request, exc: Exception) -> Response:
        # Never echo the exception into the page (it can carry paths or agent text);
        # give the human a reference that finds the traceback in the log.
        ref = _log_failure(f"handling {request.method} {request.url.path}")
        return console.error_response(
            request,
            ConsoleError(
                500,
                "internal_error",
                f"The console hit an unexpected error (reference {ref}).",
                "Reload the page to see the current state before trying again. The "
                "traceback is in the console's log under that reference.",
            ),
        )

    routes = [
        Route("/", queue, methods=["GET"]),
        Route("/reviews/{project_id}/{review_id}", review_detail, methods=["GET"]),
        Route("/reviews/{project_id}/{review_id}/decide", review_decide, methods=["POST"]),
        Route("/reviews/{project_id}/{review_id}/notes", review_note, methods=["POST"]),
        Route("/artifacts/{artifact_id}", artifact, methods=["GET"]),
        Route("/leases", leases, methods=["GET"]),
        Route("/approvals", approvals_page, methods=["GET"]),
        Route("/approvals/grant", approvals_grant, methods=["POST"]),
        Route("/approvals/revoke", approvals_revoke, methods=["POST"]),
        Route("/control", control_page, methods=["GET"]),
        Route("/control/stop", control_stop, methods=["POST"]),
        Route("/control/resume", control_resume, methods=["POST"]),
    ]
    app = Starlette(
        routes=routes,
        exception_handlers={HTTPException: http_error, Exception: server_error},
    )
    allowed = LOOPBACK_HOSTS | {name.lower() for name in config.allowed_hosts}
    return SecurityHeaders(HostGuard(app, frozenset(allowed)))


WRITE_ROUTES = (
    "/reviews/{project_id}/{review_id}/decide",
    "/reviews/{project_id}/{review_id}/notes",
    "/approvals/grant",
    "/approvals/revoke",
    "/control/stop",
    "/control/resume",
)
"""Every route that changes state. Each is POST-only."""
