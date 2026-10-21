"""``analyze.laughs`` end to end: the real ffmpeg extraction and envelope, Gemini replaced by a
loopback stub (127.0.0.1, ephemeral port) or a saved raw reply. No test touches the network.

The clip is 20 s of steady speech-band room noise at -30 dBFS RMS with a laugh-like burst at
-18 dBFS RMS from 12.237 s to 15.5 s. Gemini-style times are whole seconds, up to 1 s off.
"""

from __future__ import annotations

import hashlib
import json
import traceback
import urllib.parse
from pathlib import Path
from typing import Any

import pytest
from analyze_support import laugh_clip, needs_ffmpeg, write_wav
from gemini_stub import FILE, Stub

import aar.analyze.gemini as gemini
from aar.analyze import AnalyzeError
from aar.analyze.gemini import MAX_BODY, FileRef, UrllibGeminiTransport
from aar.analyze.laughs import (
    LAUGH_PROMPT_V1,
    PROMPT_SHA256,
    RESPONSE_SCHEMA,
    LaughSettings,
    analyse_laughs,
)
from aar.cli import main
from aar.plan.inputs import load_laughs

KEY = "test-key-0000"
ONSET = 12.237
LAUGH = {"start": "0:12", "end": "0:15", "level": 2, "kind": "laugh"}
ITEMS = [
    LAUGH,
    {"start": "0:05", "end": "0:06", "level": 1, "kind": "laugh"},  # nothing there: unconfirmed
    {"start": "0:17", "end": "0:19", "level": 3, "kind": "applause"},
    {"start": 75, "end": "1:16", "level": 2, "kind": "laugh"},  # a bare number: dropped
]


@pytest.fixture
def stub() -> Any:
    server = Stub()
    yield server
    server.close()


@pytest.fixture
def paid(stub: Stub, monkeypatch: pytest.MonkeyPatch) -> Stub:
    """The key in the environment and the module's base URL pointed at the stub."""
    monkeypatch.setenv("AAR_GEMINI_API_KEY", KEY)
    monkeypatch.setattr(gemini, "_GEMINI_BASE", stub.base)
    monkeypatch.setattr(gemini, "POLL_S", 0.01)
    stub.items = ITEMS
    return stub


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "set.wav", laugh_clip())


def cli(capsys: pytest.CaptureFixture[str], *argv: object) -> tuple[int, dict[str, Any], str]:
    code = main(["analyze", "laughs", *(str(a) for a in argv)])
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def base_args(tmp_path: Path, clip: Path) -> list[object]:
    return [clip, "--out", tmp_path / "laughs.json", "--work-dir", tmp_path / "work"]


def assert_no_key_anywhere(tmp_path: Path, *texts: str) -> None:
    needles = {KEY, urllib.parse.quote(KEY, safe=""), urllib.parse.quote_plus(KEY)}
    for text in texts:
        for needle in needles:
            assert needle not in text
    for path in tmp_path.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            for needle in needles:
                assert needle.encode() not in data, path


def exception_texts(exc: BaseException) -> list[str]:
    """Everything an exception chain can show: messages, reprs, args, notes, the traceback."""
    out: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        out += [str(current), repr(current), repr(current.args)]
        out += [str(n) for n in getattr(current, "__notes__", [])]
        out += [repr(v) for v in vars(current).values()]
        current = current.__cause__ or current.__context__
    out += traceback.format_exception(exc)
    return out


# ---------------------------------------------------------------- the paid path


@needs_ffmpeg
def test_a_paid_run_end_to_end(tmp_path, clip, paid, capsys):
    out, map_out = tmp_path / "laughs.json", tmp_path / "map.json"
    code, report, err = cli(
        capsys, *base_args(tmp_path, clip), "--allow-paid", "--map-out", map_out,
        "--offset-s", "100", "--recording", "set-1",
    )  # fmt: skip
    assert code == 0 and err == "" and report["ok"]
    assert paid.calls() == [
        "POST /upload/v1beta/files",
        "POST /upload/v1beta/files",
        f"GET /v1beta/{FILE}",
        f"GET /v1beta/{FILE}",
        "POST /v1beta/models/gemini-3.8-flash:generateContent",
        f"DELETE /v1beta/{FILE}",
    ]
    for request in paid.requests:  # the key travels in one header and nowhere else
        assert request["headers"]["x-goog-api-key"] == KEY
        assert KEY not in request["path"] + request["query"] and "key=" not in request["query"]
    flac = tmp_path / "work" / "audio.flac"
    sha = hashlib.sha256(flac.read_bytes()).hexdigest()
    start, finalize = paid.requests[0], paid.requests[1]
    assert start["headers"]["x-goog-upload-protocol"] == "resumable"
    assert start["headers"]["x-goog-upload-command"] == "start"
    assert start["headers"]["x-goog-upload-header-content-length"] == str(flac.stat().st_size)
    assert start["headers"]["x-goog-upload-header-content-type"] == "audio/flac"
    assert json.loads(start["body"]) == {"file": {"display_name": f"aar-laughs-{sha[:8]}"}}
    assert finalize["query"] == "upload_id=u1"
    assert finalize["headers"]["x-goog-upload-command"] == "upload, finalize"
    assert finalize["headers"]["x-goog-upload-offset"] == "0"
    assert finalize["body"] == flac.read_bytes()
    generate = json.loads(paid.requests[4]["body"])
    [content] = generate["contents"]
    assert content["parts"] == [
        {"file_data": {"mime_type": "audio/flac", "file_uri": f"{paid.base}/v1beta/{FILE}"}},
        {"text": LAUGH_PROMPT_V1},
    ]
    assert generate["generationConfig"] == {
        "temperature": 0,
        "responseMimeType": "application/json",
        "responseSchema": RESPONSE_SCHEMA,
    }
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema"] == "aar.laughs/1"
    header = doc["header"]
    assert header["model"] == "gemini-3.8-flash" and header["source"] == "gemini"
    assert header["prompt"] == {"version": "v1", "sha256": PROMPT_SHA256}
    assert header["audio"]["sha256"] == sha and header["audio"]["duration_s"] == 20.0
    assert header["input"]["size"] == clip.stat().st_size and header["recording"] == "set-1"
    confirmed, unconfirmed = sorted(doc["laughs"], key=lambda x: x["status"])
    assert confirmed["status"] == "confirmed" and confirmed["in_map"]
    assert abs(confirmed["file_start_s"] - ONSET) <= 0.04
    assert confirmed["start_s"] == pytest.approx(confirmed["file_start_s"] + 100, abs=1e-6)
    assert confirmed["original"] == {"file_start_s": 12.0, "file_end_s": 15.0}
    assert confirmed["bump90_db"] > 6 and confirmed["frac6"] >= 0.1
    assert abs(confirmed["file_end_s"] - 15.5) <= 0.04
    assert unconfirmed["status"] == "unconfirmed" and not unconfirmed["in_map"]
    [applause] = doc["applause"]
    assert (applause["file_start_s"], applause["level"]) == (17.0, 3)
    assert doc["drops"] == [{"index": 3, "reason": "bare-number", "raw": json.dumps(ITEMS[3])}]
    assert doc["delete"] == {"status": "ok", "file": FILE}
    assert doc["summary"]["in_map"] == 1 and doc["summary"]["returned"] == 4
    laugh_map = load_laughs(map_out)
    assert laugh_map.recording == "set-1"
    [laugh] = laugh_map.laughs
    assert (laugh.start_s, laugh.end_s, laugh.strength) == (
        confirmed["start_s"],
        confirmed["end_s"],
        0.667,
    )
    saved = json.loads((tmp_path / "work" / "raw.json").read_text(encoding="utf-8"))
    assert json.loads(saved["candidates"][0]["content"]["parts"][0]["text"]) == ITEMS
    assert_no_key_anywhere(tmp_path, json.dumps(report), err)


# ---------------------------------------------------------------- offline: --from-raw


def raw_reply(tmp_path: Path, items: Any, name: str = "raw.json") -> Path:
    text = items if isinstance(items, str) else json.dumps(items)
    path = tmp_path / name
    path.write_text(
        json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]}), encoding="utf-8"
    )
    return path


@needs_ffmpeg
@pytest.mark.parametrize(("start", "end"), [("0:12", "0:15"), ("0:13", "0:16"), ("0:12", "0:14")])
def test_the_refined_onset_lands_within_40_ms(tmp_path, clip, capsys, monkeypatch, start, end):
    monkeypatch.delenv("AAR_GEMINI_API_KEY", raising=False)
    raw = raw_reply(tmp_path, [{"start": start, "end": end, "level": 2, "kind": "laugh"}])
    code, report, _ = cli(capsys, *base_args(tmp_path, clip), "--from-raw", raw)
    assert code == 0 and report["delete"] == {"status": "skipped"}
    [laugh] = json.loads((tmp_path / "laughs.json").read_text(encoding="utf-8"))["laughs"]
    assert laugh["status"] == "confirmed"
    assert abs(laugh["file_start_s"] - ONSET) <= 0.04, laugh
    assert "onset-unrefined" not in laugh["flags"]


@needs_ffmpeg
def test_include_unconfirmed_and_applause_never_in_the_map(tmp_path, clip, capsys):
    items = [
        LAUGH,
        {"start": "0:05", "end": "0:06", "level": 1, "kind": "laugh"},  # unconfirmed
        {"start": "0:01", "end": "0:02", "level": 1, "kind": "laugh"},  # no baseline: unchecked
        {"start": "0:17", "end": "0:19", "level": 3, "kind": "applause"},
    ]
    raw = raw_reply(tmp_path, items)
    map_out = tmp_path / "map.json"
    args = [*base_args(tmp_path, clip), "--from-raw", raw, "--map-out", map_out]
    assert cli(capsys, *args)[0] == 0
    assert len(load_laughs(map_out).laughs) == 1
    assert cli(capsys, *args, "--include-unconfirmed")[0] == 0
    doc = json.loads((tmp_path / "laughs.json").read_text(encoding="utf-8"))
    statuses = sorted(x["status"] for x in doc["laughs"])
    assert statuses == ["confirmed", "unchecked", "unconfirmed"]
    unchecked = next(x for x in doc["laughs"] if x["status"] == "unchecked")
    assert unchecked["unchecked_reason"] == "no-baseline"
    laughs = load_laughs(map_out).laughs
    assert len(laughs) == 3  # the applause is never a laugh
    assert all(not (17.0 <= x.start_s < 19.0) for x in laughs)


@needs_ffmpeg
def test_overlapping_reports_of_one_laugh_merge(tmp_path, clip, capsys):
    items = [LAUGH, {"start": "0:14", "end": "0:16", "level": 3, "kind": "laugh"}]
    raw = raw_reply(tmp_path, items)
    assert cli(capsys, *base_args(tmp_path, clip), "--from-raw", raw)[0] == 0
    doc = json.loads((tmp_path / "laughs.json").read_text(encoding="utf-8"))
    [laugh] = doc["laughs"]
    assert laugh["ids"] == [0, 1] and laugh["strength"] == 1.0
    assert doc["merges"] == [
        {"into": 0, "ids": [0, 1], "file_start_s": laugh["file_start_s"],
         "file_end_s": laugh["file_end_s"]}
    ]  # fmt: skip


@needs_ffmpeg
def test_a_silent_stretch_is_unchecked(tmp_path, capsys):
    clip = write_wav(tmp_path / "quiet.wav", laugh_clip(laugh=None) * 0.0)
    raw = raw_reply(tmp_path, [LAUGH])
    assert cli(capsys, *base_args(tmp_path, clip), "--from-raw", raw)[0] == 0
    [laugh] = json.loads((tmp_path / "laughs.json").read_text(encoding="utf-8"))["laughs"]
    assert (laugh["status"], laugh["unchecked_reason"]) == ("unchecked", "silent")


@needs_ffmpeg
def test_a_reply_that_is_not_an_array_is_a_bad_response(tmp_path, clip, capsys):
    raw = raw_reply(tmp_path, "Here are the laughs I found: none.")
    code, body, _ = cli(capsys, *base_args(tmp_path, clip), "--from-raw", raw)
    assert code == 1 and body["error"]["type"] == "bad-response"
    assert "raw reply is kept" in body["error"]["hint"]


# ---------------------------------------------------------------- the cost and key guards


def test_without_allow_paid_nothing_is_called(tmp_path, clip, stub, monkeypatch, capsys):
    monkeypatch.setenv("AAR_GEMINI_API_KEY", KEY)
    monkeypatch.setattr(gemini, "_GEMINI_BASE", stub.base)
    code, body, _ = cli(capsys, *base_args(tmp_path, clip))
    assert code == 2 and body["error"]["type"] == "paid-not-allowed"
    assert "--allow-paid" in body["error"]["hint"]
    assert stub.requests == []


@pytest.mark.parametrize("value", [None, "", "   "])
def test_a_missing_or_blank_key_exits_2_naming_the_variable(
    tmp_path, clip, stub, monkeypatch, capsys, value
):
    monkeypatch.setattr(gemini, "_GEMINI_BASE", stub.base)
    if value is None:
        monkeypatch.delenv("AAR_GEMINI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("AAR_GEMINI_API_KEY", value)
    code, body, _ = cli(capsys, *base_args(tmp_path, clip), "--allow-paid")
    assert code == 2 and body["error"]["type"] == "missing-credential"
    assert "AAR_GEMINI_API_KEY" in body["error"]["message"]
    assert stub.requests == []


@needs_ffmpeg
def test_audio_longer_than_the_cap_is_refused_before_upload(tmp_path, clip, paid, capsys):
    code, body, _ = cli(capsys, *base_args(tmp_path, clip), "--allow-paid", "--max-audio-s", "19")
    assert code == 2 and body["error"]["type"] == "input-too-long"
    assert paid.requests == []


# ---------------------------------------------------------------- a hostile server


def echo_headers(status: int) -> Any:
    def route(stub: Stub, handler: Any) -> None:
        headers = dict(handler.headers.items())
        reason = f"denied for {headers.get('x-goog-api-key')}"
        stub.reply(handler, status, {"error": {"echo": headers}}, reason=reason)

    return route


@needs_ffmpeg
@pytest.mark.parametrize(
    ("route", "deleted"),
    [
        (("POST", "/v1beta/models/gemini-3.8-flash:generateContent"), True),
        (("POST", "/upload/v1beta/files"), False),
        (("GET", f"/v1beta/{FILE}"), True),
    ],
)
def test_the_key_never_leaks_from_a_hostile_error(tmp_path, clip, paid, capsys, route, deleted):
    paid.routes[route] = echo_headers(400)
    code, body, err = cli(capsys, *base_args(tmp_path, clip), "--allow-paid")
    assert code == 1 and body["error"]["type"] == "gemini-http"
    assert "[redacted]" in body["error"]["message"]  # the echo was there, and was scrubbed
    assert (f"DELETE /v1beta/{FILE}" in paid.calls()) is deleted
    assert_no_key_anywhere(tmp_path, json.dumps(body), err)
    with pytest.raises(AnalyzeError) as error:
        analyse_laughs(
            clip, tmp_path / "l.json", work_dir=tmp_path / "work", allow_paid=True,
            environ={"AAR_GEMINI_API_KEY": KEY},
        )  # fmt: skip
    texts = [*exception_texts(error.value), error.value.hint, json.dumps(error.value.to_dict())]
    assert_no_key_anywhere(tmp_path, *texts)


@needs_ffmpeg
def test_the_key_never_leaks_from_a_hostile_success(tmp_path, clip, paid, capsys):
    paid.items = [
        LAUGH,
        {"start": "0:05", "end": "0:06", "level": 2, "kind": KEY},
        {"start": f"0:0{KEY}", "end": "0:09", "level": 2, "kind": "laugh"},
    ]
    code, report, err = cli(capsys, *base_args(tmp_path, clip), "--allow-paid")
    assert code == 0
    doc = json.loads((tmp_path / "laughs.json").read_text(encoding="utf-8"))
    assert [d["reason"] for d in doc["drops"]] == ["unknown-kind", "bad-time"]
    assert "[redacted]" in doc["drops"][0]["raw"]
    assert_no_key_anywhere(tmp_path, json.dumps(report), err)


@needs_ffmpeg
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirects_are_refused_and_the_key_goes_nowhere_else(tmp_path, clip, paid, capsys, status):
    elsewhere = Stub()
    try:

        def redirect(stub: Stub, handler: Any) -> None:
            stub.reply(handler, status, {}, headers={"Location": f"{elsewhere.base}/steal"})

        paid.routes[("POST", "/v1beta/models/gemini-3.8-flash:generateContent")] = redirect
        code, body, _ = cli(capsys, *base_args(tmp_path, clip), "--allow-paid")
        assert code == 1 and body["error"]["type"] == "gemini-redirect"
        assert str(status) in body["error"]["message"]
        assert elsewhere.requests == []
        assert paid.calls()[-1] == f"DELETE /v1beta/{FILE}"  # still deleted
    finally:
        elsewhere.close()


@needs_ffmpeg
def test_an_upload_url_on_another_host_is_refused(tmp_path, clip, paid, capsys):
    elsewhere = Stub()
    try:

        def start(stub: Stub, handler: Any) -> None:
            url = f"{elsewhere.base}/upload/v1beta/files?upload_id=u1"
            stub.reply(handler, 200, {}, headers={"x-goog-upload-url": url})

        paid.routes[("POST", "/upload/v1beta/files")] = start
        code, body, _ = cli(capsys, *base_args(tmp_path, clip), "--allow-paid")
        assert code == 1 and body["error"]["type"] == "gemini-bad-reply"
        assert elsewhere.requests == []
        assert paid.calls() == ["POST /upload/v1beta/files"]  # no file, so nothing to delete
    finally:
        elsewhere.close()


@needs_ffmpeg
def test_a_failed_generate_still_deletes_the_upload(tmp_path, clip, paid, capsys):
    def fail(stub: Stub, handler: Any) -> None:
        stub.reply(handler, 500, {"error": "internal"})

    paid.routes[("POST", "/v1beta/models/gemini-3.8-flash:generateContent")] = fail
    code, body, _ = cli(capsys, *base_args(tmp_path, clip), "--allow-paid")
    assert code == 1 and "HTTP 500" in body["error"]["message"]
    assert paid.calls()[-1] == f"DELETE /v1beta/{FILE}"
    assert paid.calls().count("POST /v1beta/models/gemini-3.8-flash:generateContent") == 1


@needs_ffmpeg
def test_a_failed_delete_is_reported_not_raised(tmp_path, clip, paid, capsys):
    paid.routes[("DELETE", f"/v1beta/{FILE}")] = echo_headers(500)
    code, report, err = cli(capsys, *base_args(tmp_path, clip), "--allow-paid")
    assert code == 0
    delete = report["delete"]
    assert delete["status"] == "failed" and "HTTP 500" in delete["message"]
    doc = json.loads((tmp_path / "laughs.json").read_text(encoding="utf-8"))
    assert doc["delete"] == delete and doc["summary"]["confirmed"] == 1
    assert_no_key_anywhere(tmp_path, json.dumps(report), err)


# ---------------------------------------------------------------- the transport alone


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def transport(stub: Stub, clock: Clock | None = None) -> UrllibGeminiTransport:
    clock = clock or Clock()
    return UrllibGeminiTransport(KEY, base_url=stub.base, sleep=clock.sleep, clock=clock)


REF = FileRef(name=FILE, uri="x", mime_type="audio/flac")


def scripted(*replies: tuple[int, dict[str, str]]) -> Any:
    queue = list(replies)

    def route(stub: Stub, handler: Any) -> None:
        status, headers = queue.pop(0) if len(queue) > 1 else queue[0]
        stub.reply(handler, status, {} if status == 200 else {"error": status}, headers=headers)

    return route


@pytest.mark.parametrize(
    ("replies", "sleeps", "ok"),
    [
        ([(429, {"Retry-After": "100"}), (429, {"Retry-After": "2.5"}), (200, {})],
         [30.0, 2.5], True),
        ([(503, {}), (200, {})], [1.0], True),
        ([(429, {"Retry-After": "soon"})] * 3, [1.0, 1.0], False),
        ([(500, {}), (200, {})], [], False),
        ([(404, {}), (200, {})], [], False),
    ],
)  # fmt: skip
def test_429_and_503_are_retried_twice_with_a_bounded_retry_after(stub, replies, sleeps, ok):
    clock = Clock()
    stub.routes[("DELETE", f"/v1beta/{FILE}")] = scripted(*replies)
    t = transport(stub, clock)
    if ok:
        t.delete(REF)
    else:
        with pytest.raises(AnalyzeError) as error:
            t.delete(REF)
        assert error.value.type == "gemini-http"
    assert clock.sleeps == sleeps
    assert len(stub.requests) == len(sleeps) + 1


def test_a_reply_over_8_mib_is_refused(stub):
    def huge(stub: Stub, handler: Any) -> None:
        stub.reply(handler, 200, b"x" * (MAX_BODY + 1))

    stub.routes[("DELETE", f"/v1beta/{FILE}")] = huge
    with pytest.raises(AnalyzeError) as error:
        transport(stub).delete(REF)
    assert error.value.type == "gemini-too-large"


def test_waiting_for_active_fails_or_times_out(stub):
    stub.states = ["PROCESSING", "FAILED"]
    with pytest.raises(AnalyzeError) as failed:
        transport(stub).wait_active(REF, 60.0)
    assert failed.value.type == "gemini-file-failed"
    stub.states = ["PROCESSING"]
    clock = Clock()
    with pytest.raises(AnalyzeError) as late:
        transport(stub, clock).wait_active(REF, 5.0)
    assert late.value.type == "gemini-timeout"
    assert clock.now == 5.0 and clock.sleeps == [2.0, 2.0, 1.0]


@pytest.mark.parametrize(
    ("base", "ok"),
    [
        ("http://127.0.0.1:9", True),
        ("http://localhost:9", True),
        ("https://api.acme.invalid", True),
        ("http://api.acme.invalid", False),
        ("ftp://127.0.0.1", False),
        ("https://api.acme.invalid/?key=x", False),
    ],
)
def test_only_loopback_may_be_plain_http(base, ok):
    if ok:
        assert UrllibGeminiTransport(KEY, base_url=base).base == base
    else:
        with pytest.raises(AnalyzeError) as error:
            UrllibGeminiTransport(KEY, base_url=base)
        assert error.value.type == "invalid-argument"


def test_an_unreachable_server_is_a_typed_error(stub):
    base = stub.base
    stub.close()
    with pytest.raises(AnalyzeError) as error:
        UrllibGeminiTransport(KEY, base_url=base).delete(REF)
    assert error.value.type == "gemini-unreachable"


def test_replies_and_ids_are_validated(stub):
    def bad_file(stub: Stub, handler: Any) -> None:
        stub.reply(handler, 200, {"name": "files/../../x", "uri": "u", "mimeType": "a"})

    stub.routes[("GET", f"/v1beta/{FILE}")] = bad_file
    with pytest.raises(AnalyzeError) as error:
        transport(stub).wait_active(REF, 5.0)
    assert error.value.type == "gemini-bad-reply"
    with pytest.raises(AnalyzeError) as model:
        transport(stub).generate("../files/x", REF, "p", {})
    assert model.value.type == "invalid-argument"
    assert "test-key" not in repr(transport(stub))


def test_the_settings_are_validated():
    for bad in ({"confirm_bump_db": -1.0}, {"confirm_frac": 1.5}, {"max_audio_s": 0.0}):
        with pytest.raises(ValueError):
            LaughSettings(**bad)


def test_the_transport_scrubs_its_own_errors(stub):
    # Each layer scrubs on its own: here the transport alone, with no pipeline around it.
    stub.routes[("DELETE", f"/v1beta/{FILE}")] = echo_headers(400)
    with pytest.raises(AnalyzeError) as error:
        transport(stub).delete(REF)
    assert error.value.type == "gemini-http" and "[redacted]" in str(error.value)
    texts = exception_texts(error.value)
    assert all(KEY not in text for text in texts)


class LeakyTransport:
    """A transport that does not scrub: the pipeline must, on every path."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.deleted: list[str] = []

    def upload(self, path: Path, mime: str, display_name: str) -> FileRef:
        return REF

    def wait_active(self, ref: FileRef, timeout_s: float) -> FileRef:
        return ref

    def generate(self, model: str, ref: FileRef, prompt: str, schema: Any) -> dict[str, Any]:
        raise AnalyzeError("gemini-http", f"denied: {self.key}", hint=f"key was {self.key}")

    def delete(self, ref: FileRef) -> None:
        self.deleted.append(ref.name)
        raise RuntimeError(f"cannot delete with {self.key}")


@needs_ffmpeg
def test_the_pipeline_scrubs_what_a_transport_lets_through(tmp_path, clip):
    leaky: list[LeakyTransport] = []

    def factory(key: str) -> LeakyTransport:
        leaky.append(LeakyTransport(key))
        return leaky[-1]

    with pytest.raises(AnalyzeError) as error:
        analyse_laughs(
            clip, tmp_path / "l.json", work_dir=tmp_path / "work", allow_paid=True,
            environ={"AAR_GEMINI_API_KEY": KEY}, transport_factory=factory,
        )  # fmt: skip
    assert leaky[0].deleted == [FILE]  # the upload is deleted even though generate failed
    assert error.value.type == "gemini-http" and "[redacted]" in str(error.value)
    assert "could not be deleted" in error.value.hint
    texts = [*exception_texts(error.value), error.value.hint, json.dumps(error.value.to_dict())]
    assert_no_key_anywhere(tmp_path, *texts)
