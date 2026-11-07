"""Harness for the remote hub link tests (S15a): a real hub on loopback, and fault servers.

The node side uses only the stdlib, but the *tests* may use uvicorn to run the hub and a
small fault server to inject transport failures, slow replies, and odd responses.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from aar.hub.events import EventLog
from aar.hub.nodeapi import build_app
from aar.node.remote import RemoteConfig, RemoteFeedTransport, RemoteHubLink, _NodeClient


class ServerThread:
    """Run ``app`` under uvicorn on a free loopback port, started/stopped as a context."""

    def __init__(self, app: Starlette) -> None:
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> ServerThread:
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:  # pragma: no cover
                raise RuntimeError("uvicorn did not start within 10s")
            time.sleep(0.01)
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)

    @property
    def port(self) -> int:
        return self._server.servers[0].sockets[0].getsockname()[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def hub_app(log: EventLog) -> Starlette:
    return build_app(log)


Responder = Callable[[Request], Awaitable[Response]]


def fault_app(responder: Responder) -> Starlette:
    """A Starlette app whose single endpoint delegates to ``responder`` (any path, any method)."""

    async def entry(request: Request) -> Response:
        return await responder(request)

    return Starlette(
        routes=[
            Route(
                "/{rest:path}",
                entry,
                methods=["POST", "GET", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
            )
        ]
    )


def write_token(tmp_path: Path, token: str, *, mode: int = 0o600) -> Path:
    """Write ``token`` to a file and chmod it (0o600 by default), returning the path."""
    path = tmp_path / "token"
    path.write_text(token, encoding="utf-8")
    path.chmod(mode)
    return path


def client(base_url: str, token_file: Path, **kwargs: object) -> _NodeClient:
    return _NodeClient(RemoteConfig(hub_url=base_url, token_file=token_file, **kwargs))  # type: ignore[arg-type]


def remote_link(base_url: str, token_file: Path, **kwargs: object) -> RemoteHubLink:
    return RemoteHubLink(client(base_url, token_file, **kwargs))


def remote_transport(base_url: str, token_file: Path, **kwargs: object) -> RemoteFeedTransport:
    return RemoteFeedTransport(client(base_url, token_file, **kwargs))
