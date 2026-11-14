"""A loopback stand-in for the Gemini REST API (127.0.0.1, an ephemeral port).

It records every request (method, path, query, headers, body) except the host's bare port
probe (``is_host_probe``), and plays the happy path by default: resumable upload start and
finalize, PROCESSING then ACTIVE, a generateContent reply holding ``items`` as JSON text, and
delete. ``routes`` overrides any step: a function of ``(stub, handler)`` keyed by
``(method, path)`` (path without the query).

``tests/llm`` serves the model gateway's API from it through ``routes`` alone, with its own
``key``: a request carrying that key is never taken for a probe.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import threading
import urllib.parse
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

FILE = "files/abc123"
#: The API key the laughs tests put in the environment; the stub knows it to tell a leak.
KEY = "test-key-0000"
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/_-]{8,}={0,2}")


def _texts(request: dict[str, Any]) -> list[str]:
    body = request["body"]
    raw = body.decode("latin-1") if isinstance(body, bytes) else str(body or "")
    headers = [f"{name}: {value}" for name, value in request["headers"].items()]
    return [*headers, request["path"], request["query"], raw]


def carries_key(request: dict[str, Any], key: str = KEY) -> bool:
    """Whether a recorded request holds ``key`` anywhere (a header, the path, the query or the
    body): as is (``Authorization: Bearer`` included), URL-encoded, or base64-encoded (standard
    or URL-safe, padded or not, alone or inside a longer value such as ``Basic user:key``).
    The client's own key header counts whatever its value."""
    if "x-goog-api-key" in request["headers"]:
        return True
    secret = key.encode()
    for text in _texts(request):
        for view in (text, urllib.parse.unquote(text), urllib.parse.unquote_plus(text)):
            if key in view:
                return True
        for run in _BASE64_RUN.findall(text):
            for decode in (base64.b64decode, base64.urlsafe_b64decode):
                try:
                    if secret in decode(run + "=" * (-len(run) % 4)):
                        return True
                except (binascii.Error, ValueError):
                    continue
    return False


def is_host_probe(request: dict[str, Any], key: str = KEY) -> bool:
    """A request nobody under test made. On the awtprod host the Command Center server probes
    every new localhost listener with a bare ``GET /`` (user-agent node, b3 and traceparent
    headers), so a stub can see one at any moment, and ``requests == []`` failed 4 runs in 5 on
    dev b1a9096 (aar-evidence/s18a-val). A probe is a GET of ``/`` with no query, no body and no
    key in any form; the client's requests are POSTs and GETs of the Gemini paths with the key
    header, and anything carrying the key is always recorded."""
    return (
        request["method"] == "GET"
        and request["path"] == "/"
        and not request["query"]
        and not request["body"]
        and not carries_key(request, key)
    )


class Stub:
    def __init__(self, key: str = KEY) -> None:
        self.key = key  # the key the client under test holds
        self.requests: list[dict[str, Any]] = []
        self.items: Any = []
        self.states = ["PROCESSING", "ACTIVE"]
        self.routes: dict[tuple[str, str], Callable[[Stub, BaseHTTPRequestHandler], None]] = {}
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def handle(self) -> None:
                """A probe that resets its keep-alive connection must not print a traceback
                into the stderr the tests read; a connection that carried a real request
                still reports one."""
                self.recorded = False
                try:
                    super().handle()
                except ConnectionError:
                    if self.recorded:
                        raise

            def _serve(self) -> None:
                parts = urllib.parse.urlsplit(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                request = {
                    "method": self.command,
                    "path": parts.path,
                    "query": parts.query,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body,
                }
                if is_host_probe(request, stub.key):  # answered, never recorded
                    stub.reply(self, 404, {"error": "not a Gemini path"})
                    return
                self.recorded = True
                stub.requests.append(request)
                route = stub.routes.get((self.command, parts.path)) or stub.default
                route(stub, self)

            do_GET = do_POST = do_DELETE = _serve

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    # ------------------------------------------------------------ replies

    @staticmethod
    def reply(
        handler: BaseHTTPRequestHandler,
        status: int,
        body: Any = None,
        *,
        headers: dict[str, str] | None = None,
        reason: str | None = None,
    ) -> None:
        data = body if isinstance(body, bytes) else json.dumps(body or {}).encode("utf-8")
        handler.send_response(status, reason)
        for name, value in (headers or {}).items():
            handler.send_header(name, value)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def file(self, state: str) -> dict[str, Any]:
        return {"name": FILE, "uri": f"{self.base}/v1beta/{FILE}", "mimeType": "audio/flac",
                "state": state}  # fmt: skip

    def default(self, stub: Stub, handler: BaseHTTPRequestHandler) -> None:
        parts = urllib.parse.urlsplit(handler.path)
        method, path = handler.command, parts.path
        if method == "POST" and path == "/upload/v1beta/files" and not parts.query:
            url = f"{self.base}/upload/v1beta/files?upload_id=u1"
            self.reply(handler, 200, {}, headers={"x-goog-upload-url": url})
        elif method == "POST" and path == "/upload/v1beta/files":
            self.reply(handler, 200, {"file": self.file("PROCESSING")})
        elif method == "GET" and path == f"/v1beta/{FILE}":
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            self.reply(handler, 200, self.file(state))
        elif method == "POST" and path.endswith(":generateContent"):
            text = self.items if isinstance(self.items, str) else json.dumps(self.items)
            reply = {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}
            self.reply(handler, 200, reply)
        elif method == "DELETE" and path == f"/v1beta/{FILE}":
            self.reply(handler, 200, {})
        else:
            self.reply(handler, 404, {"error": "no such route in the stub"})

    def calls(self) -> list[str]:
        return [f"{r['method']} {r['path']}" for r in self.requests]
