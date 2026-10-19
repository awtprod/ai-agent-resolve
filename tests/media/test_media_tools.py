"""The ffmpeg/ffprobe runner: tool lookup, argument lists, timeouts, bounded output, errors."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from media_support import FFMPEG, fake_tool, needs_ffmpeg, posix_only

from aar.media import (
    MediaError,
    MediaInputError,
    MediaTimeout,
    MediaToolMissing,
    OutputTooLarge,
    input_url,
    probe,
    resolve_tool,
    run_tool,
    timeout_for,
)
from aar.media.tools import require_regular_file


def test_override_env_var_wins(monkeypatch, tmp_path):
    tool = fake_tool(tmp_path, "my-ffprobe", "exit 0")
    monkeypatch.setenv("AAR_FFPROBE", str(tool))
    assert resolve_tool("ffprobe") == str(tool)


def test_bad_override_is_reported_not_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("AAR_FFMPEG", str(tmp_path / "nope"))
    with pytest.raises(MediaToolMissing, match="AAR_FFMPEG"):
        resolve_tool("ffmpeg")


def test_missing_from_path(monkeypatch, tmp_path):
    monkeypatch.delenv("AAR_FFMPEG", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(MediaToolMissing, match="not found on PATH"):
        resolve_tool("ffmpeg")


def test_shell_strings_are_refused():
    with pytest.raises(TypeError):
        run_tool("ffmpeg", "-version")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        run_tool("ffmpeg", ["-version", 3])  # type: ignore[list-item]


def test_timeout_must_be_positive():
    with pytest.raises(ValueError):
        run_tool("ffmpeg", ["-version"], timeout_s=0)


@posix_only
def test_timeout_kills_the_whole_process_group(monkeypatch, tmp_path):
    # The child sleep keeps the pipes open; only killing the group lets the call return.
    tool = fake_tool(tmp_path, "slow", "echo starting >&2\nsleep 30 &\nwait")
    monkeypatch.setenv("AAR_FFPROBE", str(tool))
    started = time.monotonic()
    with pytest.raises(MediaTimeout) as caught:
        run_tool("ffprobe", ["x"], timeout_s=0.5)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0
    assert caught.value.timeout_s == 0.5
    assert caught.value.exit_code is None
    assert "starting" in caught.value.stderr_tail


@posix_only
def test_non_zero_exit_carries_code_and_stderr_tail(monkeypatch, tmp_path):
    tool = fake_tool(tmp_path, "boom", "echo first >&2\necho 'last words' >&2\nexit 3")
    monkeypatch.setenv("AAR_FFMPEG", str(tool))
    with pytest.raises(MediaError) as caught:
        run_tool("ffmpeg", ["x"], timeout_s=10)
    assert caught.value.exit_code == 3
    assert "last words" in caught.value.stderr_tail
    assert caught.value.summary().endswith("last words")
    result = run_tool("ffmpeg", ["x"], timeout_s=10, check=False)
    assert result.returncode == 3


@posix_only
def test_stderr_is_bounded_and_keeps_the_tail(monkeypatch, tmp_path):
    body = (
        'i=0\nwhile [ $i -lt 4000 ]; do echo "line $i padding padding padding" >&2; '
        "i=$((i+1)); done"
    )
    tool = fake_tool(tmp_path, "chatty", body)
    monkeypatch.setenv("AAR_FFMPEG", str(tool))
    result = run_tool("ffmpeg", [], timeout_s=20, stderr_limit=4096)
    assert result.stderr_truncated
    assert len(result.stderr.encode()) <= 4096
    assert result.stderr.splitlines()[-1].startswith("line 3999")
    assert result.stderr.splitlines()[0].startswith("line ")  # no partial first line


@posix_only
def test_stdout_limit_kills_and_raises(monkeypatch, tmp_path):
    tool = fake_tool(tmp_path, "flood", "while true; do echo xxxxxxxxxxxxxxxxxxxxxxxx; done")
    monkeypatch.setenv("AAR_FFMPEG", str(tool))
    started = time.monotonic()
    with pytest.raises(OutputTooLarge):
        run_tool("ffmpeg", [], timeout_s=20, stdout_limit=10_000)
    assert time.monotonic() - started < 10


@posix_only
def test_sink_errors_propagate_and_stop_the_process(monkeypatch, tmp_path):
    tool = fake_tool(tmp_path, "flood", "while true; do echo data; done")
    monkeypatch.setenv("AAR_FFMPEG", str(tool))

    def sink(_chunk: bytes) -> None:
        raise RuntimeError("sink full")

    with pytest.raises(RuntimeError, match="sink full"):
        run_tool("ffmpeg", [], timeout_s=20, stdout_sink=sink)


@posix_only
def test_unstartable_binary_is_tool_missing(monkeypatch, tmp_path):
    bad = tmp_path / "not-exec"
    bad.write_text("#!/nonexistent/interpreter\n", encoding="utf-8")
    bad.chmod(0o755)
    monkeypatch.setenv("AAR_FFMPEG", str(bad))
    with pytest.raises(MediaToolMissing, match="cannot start"):
        run_tool("ffmpeg", [], timeout_s=5)


@pytest.mark.parametrize(
    "name", ["-leading-dash.wav", "has spaces.wav", "ünïcødé ✓.wav", "a:b.wav"]
)
def test_input_url_is_absolute_and_never_an_option(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    url = input_url(name)
    assert url == "file:" + str(tmp_path / name)
    assert not url.startswith("-")


def test_require_regular_file_refuses_missing_dirs_and_fifos(tmp_path):
    with pytest.raises(MediaInputError, match="file not found"):
        require_regular_file(tmp_path / "missing.wav")
    with pytest.raises(MediaInputError, match="not a regular file"):
        require_regular_file(tmp_path)
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "pipe.wav"
        os.mkfifo(fifo)
        with pytest.raises(MediaInputError, match="not a regular file"):
            require_regular_file(fifo)


@needs_ffmpeg
def test_probe_on_a_fifo_returns_at_once(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("no FIFOs here")
    fifo = tmp_path / "pipe.mp4"
    os.mkfifo(fifo)
    started = time.monotonic()
    with pytest.raises(MediaInputError):
        probe(fifo)
    assert time.monotonic() - started < 2


@needs_ffmpeg
def test_real_tool_runs_with_an_argument_list():
    result = run_tool("ffmpeg", ["-hide_banner", "-version"], timeout_s=30)
    assert result.returncode == 0
    assert result.stdout.startswith(b"ffmpeg version")
    assert result.argv[0] == FFMPEG or Path(result.argv[0]).name.startswith("ffmpeg")


def test_timeout_for_grows_with_duration():
    assert timeout_for(None) == 60.0
    assert timeout_for(float("nan")) == 60.0
    assert timeout_for(-1) == 60.0
    assert timeout_for(100.0, per_media_s=2.0) == 260.0
