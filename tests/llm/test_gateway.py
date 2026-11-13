"""``aar.llm.gateway`` against a loopback stand-in for the model gateway: the stub of
``tests/analyze/gemini_stub.py`` (127.0.0.1, an ephemeral port), serving the chat route through
``routes``. Nothing here touches the network; the one ``live`` test is skipped unless
``AAR_LIVE=1``.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import re
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import pytest
from gemini_stub import Stub, carries_key

from aar.httpsafe import REDACTED, NoRedirectHandler, Scrubber, safe_opener
from aar.hub.lesson_drafts import (
    DEFAULT_LESSONS_MODEL,
    ReviewFeedback,
    build_messages,
    parse_drafts,
    scopes_for,
)
from aar.llm.gateway import (
    BASE_ENV,
    CHAT_PATH,
    DEFAULT_BASE_URL,
    KEY_ENV,
    MAX_BODY,
    MODELS_PATH,
    ChatReply,
    GatewayClient,
    GatewayError,
    gateway_from_env,
)

#: URL-encoding changes ``+``, ``/`` and ``=``, so the raw and the quoted forms differ.
KEY = "gw-test+key/0000=="
QUOTED = urllib.parse.quote(KEY, safe="")


def lower_hex(text: str) -> str:
    """``text`` with the hex digits of each ``%XX`` escape in lower case."""
    return re.sub(r"%[0-9A-F]{2}", lambda escape: escape.group().lower(), text)


#: Every form the scrubber claims: raw, the three URL encodings Python writes (``quote`` keeps
#: ``/`` by default), and lower-case hex escapes, which mean the same bytes.
FORMS = (
    KEY,
    QUOTED,
    urllib.parse.quote_plus(KEY),
    urllib.parse.quote(KEY),
    lower_hex(QUOTED),
    lower_hex(urllib.parse.quote(KEY)),
)
CHAT = ("POST", CHAT_PATH)
MESSAGES = [
    {"role": "system", "content": "Answer in one word."},
    {"role": "user", "content": "Ready?"},
]


def completion(content: Any = "Ready.", **fields: Any) -> dict[str, Any]:
    reply: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
    }  # fmt: skip
    reply.update(fields)
    return reply


def answer(*replies: tuple[int, Any, dict[str, str]]) -> Any:
    """A route playing ``(status, body, headers)`` replies in turn, the last one for good."""
    queue = list(replies)

    def route(stub: Stub, handler: Any) -> None:
        status, body, headers = queue.pop(0) if len(queue) > 1 else queue[0]
        stub.reply(handler, status, body, headers=headers)

    return route


class Clock:
    """A fake monotonic clock; ``sleep`` advances it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def stub() -> Any:
    server = Stub(key=KEY)
    yield server
    server.close()


def client(stub: Stub, clock: Clock | None = None, **options: Any) -> GatewayClient:
    clock = clock or Clock()
    return GatewayClient(stub.base, KEY, sleep=clock.sleep, clock=clock, **options)


def ask(gateway: GatewayClient, **options: Any) -> ChatReply:
    return gateway.chat("test-model", MESSAGES, max_tokens=64, **options)


def failure(gateway: GatewayClient) -> GatewayError:
    with pytest.raises(GatewayError) as error:
        ask(gateway)
    return error.value


def exception_texts(exc: BaseException) -> list[str]:
    """Everything an exception shows: message, repr, args, attributes, formatted traceback."""
    out = [str(exc), repr(exc), repr(exc.args), *(repr(v) for v in vars(exc).values())]
    return out + traceback.format_exception(exc)


def assert_no_key(*texts: str) -> None:
    for text in texts:
        for form in FORMS:
            assert form not in text


# ---------------------------------------------------------------- the happy path


def test_a_chat_returns_the_text_model_and_token_counts(stub):
    stub.routes[CHAT] = answer((200, completion("Keep the laugh."), {}))
    assert ask(client(stub)) == ChatReply(
        text="Keep the laugh.",
        model="test-model",
        usage={"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
    )


def test_the_request_is_a_post_with_the_key_in_its_authorization_header_only(stub):
    stub.routes[CHAT] = answer((200, completion(), {}))
    gateway = client(stub)
    ask(gateway)
    ask(gateway, temperature=0.2)
    assert stub.calls() == [f"POST {CHAT_PATH}"] * 2
    for request in stub.requests:
        assert request["headers"]["authorization"] == f"Bearer {KEY}"
        assert request["headers"]["content-type"] == "application/json"
        others = {k: v for k, v in request["headers"].items() if k != "authorization"}
        assert not carries_key({**request, "headers": others}, KEY)  # path, query, body, headers
    first, second = (json.loads(request["body"]) for request in stub.requests)
    assert first == {"model": "test-model", "messages": MESSAGES, "max_tokens": 64}
    assert second == {**first, "temperature": 0.2}


def test_model_and_usage_are_optional(stub):
    bare = completion("Fine.")
    del bare["model"], bare["usage"]
    stub.routes[CHAT] = answer(
        (200, bare, {}), (200, completion("Fine.", model=None, usage=None), {})
    )
    gateway = client(stub)
    assert ask(gateway) == ask(gateway) == ChatReply(text="Fine.", model=None, usage={})


def test_a_reply_of_exactly_1_mib_is_read_whole(stub):
    data = json.dumps(completion("Big.")).encode()
    stub.routes[CHAT] = answer((200, data + b" " * (1024 * 1024 - len(data)), {}))
    assert ask(client(stub)).text == "Big."
    assert MAX_BODY == 1_048_576


def test_an_injected_safe_opener_is_used(stub):
    seen: list[str] = []

    class Spy(urllib.request.BaseHandler):
        def http_request(self, request: urllib.request.Request) -> urllib.request.Request:
            seen.append(request.full_url)
            return request

    opener = safe_opener()
    opener.add_handler(Spy())
    stub.routes[CHAT] = answer((200, completion(), {}))
    assert ask(client(stub, opener=opener)).text == "Ready."
    assert seen == [stub.base + CHAT_PATH]


# ---------------------------------------------------------------- the model list

MODELS = ("GET", MODELS_PATH)


def test_models_lists_the_ids_the_gateway_serves(stub):
    listed = [
        {"id": "gpt-6-luna", "object": "model"},
        {"id": "two words"},  # not a model id: ignored
        {"object": "model"},
        "gpt-6-sol",
        {"id": 7},
        {"id": "gpt-6-sol"},
    ]
    stub.routes[MODELS] = answer((200, {"object": "list", "data": listed}, {}))
    assert client(stub).models() == ("gpt-6-luna", "gpt-6-sol")
    [request] = stub.requests
    assert (request["method"], request["path"], request["body"]) == ("GET", MODELS_PATH, b"")
    assert request["headers"]["authorization"] == f"Bearer {KEY}"
    assert "content-type" not in request["headers"]


@pytest.mark.parametrize(
    "body", [b"not json", [], {"data": {"id": "gpt-6-luna"}}, {"models": []}], ids=repr
)
def test_a_model_list_without_a_data_list_is_a_bad_response(stub, body):
    stub.routes[MODELS] = answer((200, body, {}))
    with pytest.raises(GatewayError) as error:
        client(stub).models()
    assert error.value.code == "bad-response"
    assert error.value.message.startswith(f"GET {MODELS_PATH}: ")


def test_a_refused_model_list_is_its_status_and_names_the_get(stub):
    stub.routes[MODELS] = answer((401, {"error": "bad key"}, {}))
    with pytest.raises(GatewayError) as error:
        client(stub).models()
    assert error.value.code == "http-401"
    assert error.value.message.startswith(f"GET {MODELS_PATH}: HTTP 401")
    assert len(stub.requests) == 1


# ---------------------------------------------------------------- retries


@pytest.mark.parametrize(
    ("retry_after", "slept"),
    [
        ("2.5", 2.5),
        ("100", 30.0),
        ("soon", 1.0),
        (None, 1.0),
        ("-3", 0.0),
        ("nan", 1.0),
        ("NaN", 1.0),
        ("inf", 30.0),
    ],
)
def test_a_429_is_retried_after_its_bounded_retry_after(stub, retry_after, slept):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    stub.routes[CHAT] = answer((429, {"error": "slow down"}, headers), (200, completion(), {}))
    clock = Clock()
    assert ask(client(stub, clock)).text == "Ready."
    assert clock.sleeps == [slept] and len(stub.requests) == 2


def test_503_is_retried_twice_then_fails(stub):
    stub.routes[CHAT] = answer((503, {"error": "busy"}, {}))
    clock = Clock()
    error = failure(client(stub, clock))
    assert error.code == "http-503"
    assert "after 2 retries" in error.message and "busy" in error.message
    assert clock.sleeps == [1.0, 1.0] and len(stub.requests) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500, 502])
def test_any_other_status_fails_at_once(stub, status):
    stub.routes[CHAT] = answer((status, {"error": "no such model"}, {}), (200, completion(), {}))
    clock = Clock()
    error = failure(client(stub, clock))
    assert error.code == f"http-{status}" and "no such model" in error.message
    assert (KEY_ENV in error.hint) is (status in (401, 403))
    assert clock.sleeps == [] and len(stub.requests) == 1


# ---------------------------------------------------------------- refusals and failures


@pytest.mark.parametrize(
    ("status", "location"),
    [(301, True), (302, True), (303, True), (307, True), (308, True), (300, True), (302, False)],
)
def test_a_redirect_is_refused_and_the_key_goes_nowhere_else(stub, status, location):
    elsewhere = Stub(key=KEY)
    try:
        headers = {"Location": f"{elsewhere.base}/steal"} if location else {}
        stub.routes[CHAT] = answer((status, {}, headers))
        error = failure(client(stub))
        assert error.code == "redirect" and str(status) in error.message
        assert elsewhere.requests == []  # it records all but the host's keyless bare probe
        assert len(stub.requests) == 1
    finally:
        elsewhere.close()


def test_a_reply_over_1_mib_is_refused(stub):
    stub.routes[CHAT] = answer((200, b"x" * (1024 * 1024 + 1), {}))
    error = failure(client(stub))
    assert error.code == "too-large" and "over 1048576 bytes" in error.message


def _deep() -> bytes:
    return b"[" * 100_000 + b"]" * 100_000


@pytest.mark.parametrize(
    ("body", "why"),
    [
        (b"<html>busy</html>", "not JSON"),
        (b"\xff\xfe{}", "not JSON"),
        (b"1" * 5000, "not JSON"),  # over Python's digit limit for int()
        (_deep(), "not JSON"),  # nested too deeply
        ([completion()], "not a JSON object"),
        ({"id": "x"}, "no choices"),
        (completion(choices=[]), "no choices"),
        (completion(choices="none"), "no choices"),
        (completion(choices=["Ready."]), "no message object"),
        (completion(choices=[{"message": "Ready."}]), "no message object"),
        (completion(None), "content is not a string"),
        (completion(["Ready."]), "content is not a string"),
        (completion(""), "content is empty"),
        (completion(" \n\t"), "content is empty"),
        (completion(model=7), "model is not a string"),
        (completion(usage="lots"), "usage is not an object"),
        (completion(usage={"total_tokens": "14"}), "usage.total_tokens"),
        (completion(usage={"prompt_tokens": -1}), "usage.prompt_tokens"),
        (completion(usage={"completion_tokens": True}), "usage.completion_tokens"),
        (completion(usage={"total_tokens": 1.5}), "usage.total_tokens"),
    ],
)
def test_a_malformed_reply_is_a_bad_response(stub, body, why):
    stub.routes[CHAT] = answer((200, body, {}))
    error = failure(client(stub))
    assert error.code == "bad-response" and why in error.message


def test_a_gateway_that_does_not_answer_in_time_is_a_timeout(stub):
    release = threading.Event()

    def stall(stub: Stub, handler: Any) -> None:
        release.wait(10)  # no reply; the connection closes when this returns

    stub.routes[CHAT] = stall
    clock = Clock()
    try:
        error = failure(GatewayClient(stub.base, KEY, timeout_s=0.3, sleep=clock.sleep))
    finally:
        release.set()
    assert error.code == "timeout" and "0.3 s" in error.message
    assert clock.sleeps == [] and len(stub.requests) == 1  # never retried


def test_a_trickled_reply_cannot_outlast_the_deadline(stub):
    """Each byte arrives well inside the socket timeout; the whole reply would take seconds."""
    stop = threading.Event()

    def trickle(stub: Stub, handler: Any) -> None:
        data = json.dumps(completion()).encode()
        handler.send_response(200)
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        for index in range(len(data)):
            if stop.wait(0.05):
                return
            try:
                handler.wfile.write(data[index : index + 1])
            except OSError:  # the client hung up, as it should
                return

    stub.routes[CHAT] = trickle
    started = time.monotonic()
    try:
        error = failure(GatewayClient(stub.base, KEY, timeout_s=0.5))
    finally:
        stop.set()
    assert error.code == "timeout"
    assert time.monotonic() - started < 3  # the whole reply needs 0.05 s x its ~250 bytes


def test_an_unreachable_gateway_is_a_network_error():
    closed = Stub(key=KEY)
    base = closed.base
    closed.close()
    error = failure(GatewayClient(base, KEY))
    assert error.code == "network" and "nothing was retried" in error.hint


@pytest.mark.parametrize(
    "base",
    [
        "http://127.0.0.1:8317",
        "http://localhost:8317/",
        "http://[::1]:8317",
        "https://gw.example.invalid",
        "https://gw.example.invalid:8443/prefix",
    ],
)
def test_https_or_a_loopback_base_is_accepted(base):
    assert GatewayClient(base, KEY).base_url == base.rstrip("/")


@pytest.mark.parametrize(
    "base",
    [
        "http://gw.example.invalid",  # plain http off loopback
        "http://10.0.0.5:8317",
        "ftp://127.0.0.1:8317",
        "127.0.0.1:8317",
        "",
        "https://gw.example.invalid/?k=1",
        "https://gw.example.invalid/#k",
        "https://user:secret@gw.example.invalid",
        "http://127.0.0.1:99999",
        "http://127.0.0.1:port",
        f"https://gw.example.invalid/{KEY}",  # the key goes in the header and nowhere else
        f"https://gw.example.invalid/?{QUOTED}",
        *(f"https://gw.example.invalid/prefix/{form}" for form in FORMS),
        f"http://127.0.0.1:8317/{urllib.parse.quote(KEY)}",  # quote() keeps "/"
        f"http://127.0.0.1:8317/{lower_hex(QUOTED)}",
        "http://127.0.0.1:8317/gw-test%2Bkey%2f0000%3d%3D",  # escapes in mixed case
    ],
)
def test_a_refused_base_url_is_an_origin_error(base):
    with pytest.raises(GatewayError) as error:
        GatewayClient(base, KEY)
    assert error.value.code == "origin" and BASE_ENV in error.value.message
    assert_no_key(*exception_texts(error.value))


def test_a_request_off_the_base_origin_is_refused_before_it_is_sent(stub):
    elsewhere = Stub(key=KEY)
    try:
        with pytest.raises(GatewayError) as error:
            client(stub)._post(elsewhere.base + CHAT_PATH, b"{}")
        assert error.value.code == "origin"
        assert elsewhere.requests == [] and stub.requests == []
    finally:
        elsewhere.close()


@pytest.mark.parametrize(
    "opener",
    [
        urllib.request.build_opener(),  # urllib's own: follows redirects, reads proxy variables
        urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": "http://proxy.example.invalid:3128"}),
            NoRedirectHandler(),
        ),
        object(),
    ],
)
def test_an_opener_that_could_leak_the_key_is_refused(opener):
    with pytest.raises(ValueError, match="opener"):
        GatewayClient(DEFAULT_BASE_URL, KEY, opener=opener)


def test_a_safe_opener_with_a_redirect_handler_added_is_refused():
    opener = safe_opener()
    opener.add_handler(urllib.request.HTTPRedirectHandler())
    with pytest.raises(ValueError, match="redirects"):
        GatewayClient(DEFAULT_BASE_URL, KEY, opener=opener)


@pytest.mark.parametrize(
    "change",
    [
        {"model": ""},
        {"model": "two words"},
        {"model": "m" * 129},
        {"messages": []},
        {"messages": "Ready?"},
        {"messages": [{"role": "user"}]},
        {"messages": [{"role": "", "content": "Ready?"}]},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"max_tokens": 1.5},
        {"temperature": float("nan")},
        {"temperature": 2.5},
        {"temperature": True},
    ],
)
def test_a_bad_argument_raises_before_anything_is_sent(stub, change):
    call = {"model": "test-model", "messages": MESSAGES, "max_tokens": 64, **change}
    with pytest.raises(ValueError):
        client(stub).chat(call.pop("model"), call.pop("messages"), **call)
    assert stub.requests == []


@pytest.mark.parametrize("timeout_s", [0, -1, float("inf"), "60"])
def test_the_timeout_must_be_a_positive_number(timeout_s):
    with pytest.raises(ValueError):
        GatewayClient(DEFAULT_BASE_URL, KEY, timeout_s=timeout_s)


# ---------------------------------------------------------------- the key


def echo(status: int) -> Any:
    """A hostile gateway: the key comes back raw, URL-quoted, in the echoed request headers and
    in the reason phrase."""

    def route(stub: Stub, handler: Any) -> None:
        headers = dict(handler.headers.items())
        body = {"forms": list(FORMS), "echo": headers}
        stub.reply(handler, status, body, reason=f"denied for {headers.get('Authorization')}")

    return route


def test_the_key_never_leaks_from_a_hostile_error(stub, caplog):
    caplog.set_level(logging.DEBUG, logger="aar.llm.gateway")
    stub.routes[CHAT] = echo(429)  # retried twice (logged), then an error
    gateway = client(stub)
    error = failure(gateway)
    assert error.code == "http-429" and "[redacted]" in error.message  # echoed, then scrubbed
    assert error.__cause__ is None and error.__context__ is None
    assert len(caplog.records) == 5  # three attempts and two retries
    logged = [record.getMessage() for record in caplog.records]
    assert_no_key(*exception_texts(error), error.hint, repr(gateway), caplog.text, *logged)


def test_the_key_never_leaks_from_a_transport_failure_that_quotes_the_request(stub):
    """Whatever text reaches an error is scrubbed last, even a transport's own message."""

    class Leaky(urllib.request.BaseHandler):
        handler_order = 100  # before the real HTTP handler: nothing is sent

        def http_open(self, request: urllib.request.Request) -> None:
            raise urllib.error.URLError(f"refused {request.get_header('Authorization')} {QUOTED}")

    opener = safe_opener()
    opener.add_handler(Leaky())
    error = failure(client(stub, opener=opener))
    assert error.code == "network" and "[redacted]" in error.message
    assert error.__cause__ is None and error.__context__ is None
    assert_no_key(*exception_texts(error))
    assert stub.requests == []


def test_the_key_never_leaks_from_a_hostile_reply(stub):
    stub.routes[CHAT] = answer((200, completion(f"Echo {' '.join(FORMS)}", model=f"m-{KEY}"), {}))
    reply = ask(client(stub))
    assert reply.text == "Echo" + f" {REDACTED}" * len(FORMS) and reply.model == "m-[redacted]"
    assert_no_key(repr(reply))


@pytest.mark.parametrize("status", [400, 401, 500])
def test_every_form_of_the_key_in_an_error_body_is_scrubbed(stub, status):
    stub.routes[CHAT] = answer(
        (status, {name: form for name, form in enumerate(map(str, FORMS))}, {})
    )
    error = failure(client(stub))
    assert error.message.count(REDACTED) >= len(FORMS)
    assert_no_key(*exception_texts(error), error.hint)


def test_the_scrubber_matches_escapes_in_either_case_and_nothing_else():
    scrub = Scrubber("Ab+c/=")
    for form in (
        "Ab+c/=",
        "Ab%2Bc%2F%3D",
        "Ab%2Bc/%3D",
        "Ab%2bc%2f%3d",
        "Ab%2Bc%2f%3D",
        "Ab%2bc/%3d",
    ):
        assert scrub(f"<{form}>") == f"<{REDACTED}>", form
    for other in ("ab+c/=", "AB%2BC%2F%3D", "Ab%2Bc", "Ab+c/"):  # not the key: left alone
        assert scrub(other) == other
    assert Scrubber("")("text") == "text"


def test_proxy_environment_variables_are_ignored(stub, monkeypatch):
    """Every proxy variable points at a second stub, which must receive nothing."""
    proxy = Stub(key=KEY)
    try:
        for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy"):
            monkeypatch.setenv(name, proxy.base)
        monkeypatch.setenv("ALL_PROXY", proxy.base)
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.delenv("NO_PROXY", raising=False)
        # The variables do route urllib's own opener through the proxy here.
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.build_opener().open(stub.base + "/through-the-proxy", timeout=5)
        assert [request["path"] for request in proxy.requests] == ["/through-the-proxy"]
        proxy.requests.clear()
        stub.routes[MODELS] = answer((200, {"data": [{"id": "test-model"}]}, {}))
        stub.routes[CHAT] = answer((200, completion(), {}))
        gateway = gateway_from_env({KEY_ENV: KEY, BASE_ENV: stub.base})
        assert gateway.models() == ("test-model",) and ask(gateway).text == "Ready."
        assert proxy.requests == [] and len(stub.requests) == 2
    finally:
        proxy.close()


@pytest.mark.parametrize("before", [b"x" * 290, b"x" + b" " * 2040])
def test_no_part_of_a_key_cut_off_by_the_excerpt_shows(stub, before):
    """The key straddles the 300-character excerpt; or it straddles the 2048-byte read, with
    whitespace before it that collapsing would pull into view."""
    stub.routes[CHAT] = answer((400, before + KEY.encode() + b" end", {}))
    error = failure(client(stub))
    assert error.code == "http-400" and KEY[:6] not in error.message


def test_an_error_body_cut_short_by_the_deadline_is_not_shown(stub):
    """The key trickles in a byte at a time, each well inside the socket timeout, so the read
    stops at the deadline with part of the key in hand: none of that read may show."""
    stop = threading.Event()

    def trickle_the_key(stub: Stub, handler: Any) -> None:
        handler.send_response(400)
        handler.send_header("Content-Length", str(290 + len(KEY)))
        handler.end_headers()
        try:
            handler.wfile.write(b"x" * 290)
            for index in range(len(KEY)):
                if stop.wait(0.1):
                    return
                handler.wfile.write(KEY.encode()[index : index + 1])
        except OSError:  # the client hung up, as it should
            return

    stub.routes[CHAT] = trickle_the_key
    try:
        error = failure(GatewayClient(stub.base, KEY, timeout_s=0.5))
    finally:
        stop.set()
    assert error.code == "http-400"
    assert "x" * 10 not in error.message and KEY[:2] not in error.message


@pytest.mark.parametrize("key", ["", "   ", "two words", "line\nbreak", "kéy", "a=b"])
def test_a_key_that_is_not_a_bearer_token_is_not_configured(key):
    with pytest.raises(GatewayError) as error:
        GatewayClient(DEFAULT_BASE_URL, key)
    assert error.value.code == "not_configured" and KEY_ENV in error.value.message
    if key.strip():
        assert key.strip() not in "\n".join(exception_texts(error.value))


def test_the_stub_records_a_bare_probe_that_carries_this_suites_key(stub):
    """With ``Stub(key=KEY)`` a keyed GET / is recorded, never dropped as the host's probe, so
    ``requests == []`` above still means that nothing reached the stub."""
    connection = http.client.HTTPConnection("127.0.0.1", stub.port, timeout=5)
    try:
        connection.request("GET", "/", headers={"Authorization": f"Bearer {KEY}"})
        connection.getresponse().read()
    finally:
        connection.close()
    assert len(stub.requests) == 1


# ---------------------------------------------------------------- gateway_from_env


def test_gateway_from_env_defaults_to_the_local_gateway(stub):
    assert gateway_from_env({KEY_ENV: KEY}).base_url == DEFAULT_BASE_URL == "http://127.0.0.1:8317"
    assert gateway_from_env({KEY_ENV: KEY, BASE_ENV: "  "}).base_url == DEFAULT_BASE_URL
    gateway = gateway_from_env({KEY_ENV: f" {KEY}\n", BASE_ENV: f"{stub.base}/"})
    assert gateway.base_url == stub.base
    stub.routes[CHAT] = answer((200, completion(), {}))
    ask(gateway)
    assert stub.requests[0]["headers"]["authorization"] == f"Bearer {KEY}"


def test_gateway_from_env_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv(KEY_ENV, KEY)
    monkeypatch.delenv(BASE_ENV, raising=False)
    assert gateway_from_env().base_url == DEFAULT_BASE_URL


@pytest.mark.parametrize(
    "environ",
    [{}, {KEY_ENV: ""}, {KEY_ENV: " \n"}, {BASE_ENV: "https://gw.example.invalid"}],
)
def test_gateway_from_env_without_a_key_names_the_variable_and_no_value(environ):
    with pytest.raises(GatewayError) as error:
        gateway_from_env(environ)
    assert error.value.code == "not_configured" and KEY_ENV in error.value.message
    assert "gw.example.invalid" not in error.value.message + error.value.hint


# ---------------------------------------------------------------- live (AAR_LIVE=1)


@pytest.mark.live
def test_live_the_default_lessons_model_drafts_through_the_gateway():
    """One small real call. ``gateway_from_env`` takes the key from the environment; this test
    never reads or prints it."""
    if not os.environ.get(KEY_ENV, "").strip():
        pytest.skip(f"{KEY_ENV} is not set")
    feedback = ReviewFeedback(
        project_id="prj_live",
        review_id="rev_live",
        title="Cut 1",
        decision="reject",
        reason="The ending runs on after the big laugh.",
        notes=[("00:41:10:00", "End here, on the laugh."), ("00:41:30:00", "The applause drags.")],
        show="Synthetic Show",
        client=None,
    )
    messages = build_messages(feedback, {})
    reply = gateway_from_env().chat(DEFAULT_LESSONS_MODEL, messages, max_tokens=2000)
    result = parse_drafts(reply.text, scopes=scopes_for(feedback))
    assert len(result.drafts) <= 3
