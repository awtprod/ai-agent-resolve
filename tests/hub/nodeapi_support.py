"""Run the node API under a real uvicorn server on loopback, in a thread (S15a tests).

A :class:`ServerThread` binds port 0 (the OS picks a free port), starts uvicorn in a daemon
thread, waits for it to come up and shuts it down cleanly. :func:`post` is a tiny stdlib HTTP
client, so the hub-side tests never depend on the node-side client.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import uvicorn
from starlette.applications import Starlette


class ServerThread:
    """A uvicorn server for ``app`` on a free loopback port, started/stopped as a context."""

    def __init__(self, app: Starlette) -> None:
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> ServerThread:
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:  # pragma: no cover - startup failure
                raise RuntimeError("uvicorn did not start within 10s")
            time.sleep(0.01)
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)

    @property
    def base_url(self) -> str:
        sock = self._server.servers[0].sockets[0]
        port = sock.getsockname()[1]
        return f"http://127.0.0.1:{port}"


@dataclass(frozen=True)
class Resp:
    status: int
    data: dict[str, Any]


def post(
    base_url: str,
    op: str,
    body: dict[str, Any] | None = None,
    *,
    token: str | None = None,
    method: str = "POST",
    content_type: str | None = "application/json",
    raw: bytes | None = None,
    extra_headers: dict[str, str] | None = None,
) -> Resp:
    """POST ``body`` as JSON to ``/node/v1/<op>`` and return the status and parsed JSON."""
    url = f"{base_url}/node/v1/{op}"
    payload = raw if raw is not None else json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method=method)
    if content_type is not None:
        req.add_header("Content-Type", content_type)
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    for key, value in (extra_headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return Resp(resp.status, json.loads(resp.read().decode("utf-8")))
    except urllib.error.HTTPError as exc:
        return Resp(exc.code, json.loads(exc.read().decode("utf-8")))
