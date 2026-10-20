"""A loopback stand-in for the Gemini REST API (127.0.0.1, an ephemeral port).

It records every request (method, path, query, headers, body) and plays the happy path by
default: resumable upload start and finalize, PROCESSING then ACTIVE, a generateContent reply
holding ``items`` as JSON text, and delete. ``routes`` overrides any step: a function of
``(stub, handler)`` keyed by ``(method, path)`` (path without the query).
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

FILE = "files/abc123"


class Stub:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.items: Any = []
        self.states = ["PROCESSING", "ACTIVE"]
        self.routes: dict[tuple[str, str], Callable[[Stub, BaseHTTPRequestHandler], None]] = {}
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _serve(self) -> None:
                parts = urllib.parse.urlsplit(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                stub.requests.append(
                    {
                        "method": self.command,
                        "path": parts.path,
                        "query": parts.query,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    }
                )
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
