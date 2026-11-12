"""S19d: ``aar analyze transcript-text`` and the wordless stretches of a transcript."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aar.analyze.text import (
    NO_WORDS,
    Stretch,
    render_text,
    text_lines,
    wordless_stretches,
)
from aar.cli import main
from aar.plan.inputs import Transcript, Word


def words(*spec: tuple[str, float, float]) -> list[Word]:
    return [Word(text=t, start_s=a, end_s=b) for t, a, b in spec]


def transcript(*spec: tuple[str, float, float], recording: str | None = "show") -> Transcript:
    return Transcript(recording=recording, words=words(*spec))


# ---------------------------------------------------------------- wordless stretches


def test_wordless_stretches_between_words_and_at_the_ends() -> None:
    w = words(("a", 70.0, 70.5), ("b", 70.4, 71.0), ("c", 131.0, 131.5), ("d", 140.0, 141.0))
    # 71.0 -> 131.0 is exactly 60 s: it counts. Overlapping words (a, b) leave no hole.
    assert wordless_stretches(w, 60.0) == [Stretch(71.0, 131.0)]
    # With the file's bounds, the head (0 -> 70) and the tail (141 -> 300) count too.
    assert wordless_stretches(w, 60.0, start_s=0.0, end_s=300.0) == [
        Stretch(0.0, 70.0),
        Stretch(71.0, 131.0),
        Stretch(141.0, 300.0),
    ]
    assert wordless_stretches(w, 60.01) == []
    # The tail and the head exactly min_s long count too.
    assert wordless_stretches(w, 60.0, start_s=10.0, end_s=201.0) == [
        Stretch(10.0, 70.0),
        Stretch(71.0, 131.0),
        Stretch(141.0, 201.0),
    ]
    assert wordless_stretches(w, 60.0, start_s=10.01, end_s=200.99) == [Stretch(71.0, 131.0)]
    # A long word covering a gap hides it, whatever the order the words come in.
    covered = [*w, Word(text="hum", start_s=60.0, end_s=135.0)]
    assert wordless_stretches(list(reversed(covered)), 60.0) == []


def test_no_words_is_one_stretch_over_the_whole_file() -> None:
    assert wordless_stretches([], 60.0, start_s=5.0, end_s=95.0) == [Stretch(5.0, 95.0)]
    assert wordless_stretches([], 60.0) == []
    assert Stretch(5.0, 95.0).duration_s == 90.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_wordless_min_must_be_positive_and_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="min_s"):
        wordless_stretches([], bad)


# ---------------------------------------------------------------- text


def test_one_line_per_run_with_a_break_at_each_gap() -> None:
    t = transcript(
        ("so", 612.4, 612.6),
        ("anyway", 612.65, 613.0),
        ("the", 613.6, 613.7),  # 0.6 s gap: same line
        ("dog", 614.5, 614.9),  # 0.8 s gap: new line
        ("got", 615.0, 615.2),
    )
    assert render_text(t) == "[0612.4-0613.7] so anyway the\n[0614.5-0615.2] dog got\n"


def test_the_gap_is_strictly_longer_than_gap_s() -> None:
    t = transcript(("a", 1.0, 1.5), ("b", 2.0, 2.5), ("c", 3.01, 3.5))
    assert [line.text for line in text_lines(t, gap_s=0.5)] == ["a b", "c"]


def test_a_wordless_stretch_is_its_own_line() -> None:
    t = transcript(("hello", 1200.0, 1205.0), ("welcome", 2890.4, 2891.0), ("back", 2891.1, 2893.0))
    assert render_text(t) == (
        f"[1200.0-1205.0] hello\n[1205.0-2890.4] {NO_WORDS}\n[2890.4-2893.0] welcome back\n"
    )
    # Under the threshold it is just a line break.
    assert render_text(t, wordless_min_s=2000.0) == (
        "[1200.0-1205.0] hello\n[2890.4-2893.0] welcome back\n"
    )


def test_a_line_breaks_before_it_passes_max_chars() -> None:
    spec = [(f"w{i:02d}", i * 0.3, i * 0.3 + 0.2) for i in range(30)]
    lines = text_lines(transcript(*spec), max_chars=20)
    assert all(len(line.text) <= 20 for line in lines)
    assert [line.text for line in lines][:2] == ["w00 w01 w02 w03 w04", "w05 w06 w07 w08 w09"]
    assert " ".join(line.text for line in lines).split() == [w for w, _, _ in spec]
    # Each line's times are its own words' span.
    assert (lines[1].start_s, lines[1].end_s) == pytest.approx((1.5, 2.9))
    # A word longer than the limit is a line of its own, never split.
    long = "x" * 30
    got = text_lines(transcript(("a", 0, 1), (long, 1, 2)), max_chars=20)
    assert [line.text for line in got] == ["a", long]


def test_the_line_cap_counts_the_joining_space() -> None:
    # 9 + 1 + 10 = 20 fits a 20-character line; 9 + 1 + 11 = 21 does not.
    fits = transcript(("a" * 9, 0.0, 0.1), ("b" * 10, 0.2, 0.3))
    assert [line.text for line in text_lines(fits, max_chars=20)] == ["a" * 9 + " " + "b" * 10]
    over = transcript(("a" * 9, 0.0, 0.1), ("b" * 11, 0.2, 0.3))
    assert [line.text for line in text_lines(over, max_chars=20)] == ["a" * 9, "b" * 11]


def test_a_line_break_keeps_the_furthest_covered_time() -> None:
    # A long word [0, 3] fills the line; [1, 1.2] breaks onto a new line; [2, 2.2] is still
    # inside [0, 3], so there is no gap and no wordless line (Prism LOW).
    t = transcript(("x" * 20, 0.0, 3.0), ("a", 1.0, 1.2), ("b", 2.0, 2.2), ("c", 3.5, 3.6))
    lines = text_lines(t, max_chars=20, wordless_min_s=0.5)
    assert [(line.text, line.wordless) for line in lines] == [
        ("x" * 20, False),
        ("a b", False),
        (NO_WORDS, True),
        ("c", False),
    ]
    # The only hole is after the furthest end reached (3.0), not after the line's own end.
    assert (lines[2].start_s, lines[2].end_s) == (3.0, 3.5)
    # Nor does the hidden gap break a line on gap_s.
    assert [x.text for x in text_lines(t, max_chars=20, gap_s=0.5)] == ["x" * 20, "a b c"]


def test_gap_boundaries_are_exact() -> None:
    # 614.5 + 0.7 - 614.5 is 0.70000000000005 in floats: still "not longer than" gap_s.
    t = transcript(("a", 614.0, 614.5), ("b", 614.5 + 0.7, 615.5))
    assert [x.text for x in text_lines(t, gap_s=0.7)] == ["a b"]
    t = transcript(("a", 614.0, 614.5), ("b", 614.5 + 0.7 + 2e-6, 615.5))
    assert [x.text for x in text_lines(t, gap_s=0.7)] == ["a", "b"]
    # A gap of exactly wordless_min_s is a wordless line, float noise or not.
    for end, start in ((10.0, 70.0), (0.1 + 0.2, 0.1 + 0.2 + 60.0)):
        t = transcript(("a", end - 0.1, end), ("b", start, start + 0.1))
        assert [x.wordless for x in text_lines(t)] == [False, True, False]
    t = transcript(("a", 9.9, 10.0), ("b", 69.99, 70.0))
    assert [x.wordless for x in text_lines(t)] == [False, False]


def test_control_characters_never_reach_the_text() -> None:
    t = transcript(
        ("a\x00b\x1b[31mc\x7f\x9bd", 1.0, 1.4), ("\x07", 1.5, 1.6), ("e\u2028f", 1.7, 2.0)
    )
    out = render_text(t)
    assert out == "[0001.0-0002.0] ab[31mcd e f\n"
    assert not any((ord(c) < 32 and c != "\n") or 127 <= ord(c) < 160 for c in out)


@pytest.mark.parametrize("bad", [float("inf"), -float("inf"), 1e300, 1e7 + 1])
def test_absurd_times_are_refused(bad: float) -> None:
    t = Transcript.model_construct(
        recording=None, words=[Word.model_construct(text="a", start_s=1.0, end_s=bad)]
    )
    with pytest.raises(ValueError, match="finite number of seconds"):
        render_text(t)


def test_the_largest_allowed_time_still_renders() -> None:
    t = transcript(("a", 1.0, 2.0), ("b", 1e7 - 1, 1e7))
    assert render_text(t, wordless_min_s=1e9) == (
        "[00000001.0-00000002.0] a\n[09999999.0-10000000.0] b\n"
    )


def test_whitespace_inside_a_word_never_breaks_a_line() -> None:
    t = transcript(("one\ntwo", 1.0, 1.4), ("  \t ", 1.5, 1.6), ("three\tfour", 1.7, 2.0))
    assert render_text(t) == "[0001.0-0002.0] one two three four\n"


def test_wide_times_pad_to_one_width_and_overlaps_extend_the_line() -> None:
    t = transcript(("a", 5.0, 9.0), ("b", 5.5, 6.0), ("c", 67851.25, 67851.5))
    assert render_text(t, wordless_min_s=1e9) == ("[00005.0-00009.0] a b\n[67851.2-67851.5] c\n")


def test_unsorted_input_and_determinism() -> None:
    spec = [("a", 1.0, 1.2), ("b", 1.3, 1.5), ("c", 9.0, 9.2)]
    one = render_text(Transcript(recording=None, words=words(*spec)))
    two = render_text(Transcript.model_construct(recording=None, words=words(*reversed(spec))))
    assert one == two == "[0001.0-0001.5] a b\n[0009.0-0009.2] c\n"
    assert render_text(transcript()) == ""


@pytest.mark.parametrize(
    "kwargs",
    [{"gap_s": 0.0}, {"gap_s": float("nan")}, {"max_chars": 19}, {"max_chars": True},
     {"wordless_min_s": -1.0}],
)  # fmt: skip
def test_bad_settings_are_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        render_text(transcript(("a", 0.0, 1.0)), **kwargs)


# ---------------------------------------------------------------- CLI


def run(capsys: pytest.CaptureFixture[str], *argv: object) -> tuple[int, dict[str, Any]]:
    code = main(["analyze", "transcript-text", *(str(a) for a in argv)])
    return code, json.loads(capsys.readouterr().out)


def write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


SAMPLE = {
    "recording": "show",
    "words": [
        {"text": "hello", "start_s": 10.0, "end_s": 10.4},
        {"text": "there", "start_s": 10.5, "end_s": 10.9},
        {"text": "later", "start_s": 100.0, "end_s": 100.5},
    ],
}


def test_cli_writes_the_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src, out = write(tmp_path / "t.json", SAMPLE), tmp_path / "t.txt"
    code, report = run(capsys, src, "--out", out)
    assert code == 0 and report["ok"]
    assert out.read_text(encoding="utf-8") == (
        f"[0010.0-0010.9] hello there\n[0010.9-0100.0] {NO_WORDS}\n[0100.0-0100.5] later\n"
    )
    assert report["lines"] == 3 and report["wordless_lines"] == 1 and report["words"] == 3
    assert report["settings"] == {"gap_s": 0.7, "max_chars": 120, "wordless_min_s": 60.0}
    code, report = run(capsys, src, "--out", tmp_path / "u.txt", "--wordless-min-s", "100",
                       "--gap-s", "0.05", "--max-chars", "40")  # fmt: skip
    assert code == 0 and (tmp_path / "u.txt").read_text(encoding="utf-8") == (
        "[0010.0-0010.4] hello\n[0010.5-0010.9] there\n[0100.0-0100.5] later\n"
    )


def test_cli_never_overwrites_silently(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src, out = write(tmp_path / "t.json", SAMPLE), tmp_path / "t.txt"
    out.write_text("keep me", encoding="utf-8")
    code, body = run(capsys, src, "--out", out)
    assert code == 2 and body["error"]["type"] == "output-exists"
    assert "--force" in body["error"]["hint"] and out.read_text(encoding="utf-8") == "keep me"
    code, _ = run(capsys, src, "--out", out, "--force")
    assert code == 0 and out.read_text(encoding="utf-8").startswith("[0010.0-0010.9] hello")


def test_cli_accepts_a_bare_word_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = write(tmp_path / "t.json", SAMPLE["words"])
    code, report = run(capsys, src, "--out", tmp_path / "t.txt")
    assert code == 0 and report["recording"] is None


@pytest.mark.parametrize(
    ("content", "kind"),
    [
        ("not json", "invalid-transcript"),
        (json.dumps({"words": [{"text": "a", "start_s": 2, "end_s": 1}]}), "invalid-transcript"),
        (json.dumps({"words": [], "extra": 1}), "invalid-transcript"),
        (None, "input-missing"),
    ],
)
def test_cli_validates_the_transcript(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str | None, kind: str
) -> None:
    src = tmp_path / "t.json"
    if content is not None:
        src.write_text(content, encoding="utf-8")
    code, body = run(capsys, src, "--out", tmp_path / "t.txt")
    assert code == 2 and body["error"]["type"] == kind
    assert not (tmp_path / "t.txt").exists()


@pytest.mark.parametrize(
    "argv", [["--gap-s", "0"], ["--max-chars", "5"], ["--wordless-min-s", "nan"]]
)
def test_cli_refuses_bad_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    src = write(tmp_path / "t.json", SAMPLE)
    code, body = run(capsys, src, "--out", tmp_path / "t.txt", *argv)
    assert code == 2 and body["error"]["type"] == "invalid-argument"
    assert not (tmp_path / "t.txt").exists()


def test_cli_refuses_an_output_in_a_missing_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = write(tmp_path / "t.json", SAMPLE)
    code, body = run(capsys, src, "--out", tmp_path / "nope" / "t.txt")
    assert code == 2 and body["error"]["type"] == "invalid-argument"


def test_cli_counts_wordless_lines_from_the_structure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = {
        "words": [
            {"text": "he said", "start_s": 1.0, "end_s": 1.5},
            {"text": NO_WORDS, "start_s": 1.6, "end_s": 2.0},  # spoken, not a hole
            {"text": "later", "start_s": 100.0, "end_s": 100.5},
        ]
    }
    code, report = run(capsys, write(tmp_path / "t.json", data), "--out", tmp_path / "t.txt")
    assert code == 0 and report["wordless_lines"] == 1 and report["lines"] == 3


@pytest.mark.parametrize("raw", ["Infinity", "1e300", "-Infinity"])
def test_cli_refuses_absurd_times(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], raw: str
) -> None:
    src = tmp_path / "t.json"
    src.write_text(
        '{"words": [{"text": "a", "start_s": 1.0, "end_s": 2.0},'
        f' {{"text": "b", "start_s": 3.0, "end_s": {raw}}}]}}',
        encoding="utf-8",
    )
    code, body = run(capsys, src, "--out", tmp_path / "t.txt")
    assert code == 2 and body["error"]["type"] == "invalid-transcript"
    assert not (tmp_path / "t.txt").exists()


@pytest.mark.parametrize("force", [[], ["--force"]])
@pytest.mark.parametrize("how", ["same", "symlink", "relative"])
def test_cli_never_writes_over_its_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    force: list[str],
    how: str,
) -> None:
    src = write(tmp_path / "t.json", SAMPLE)
    before = src.read_bytes()
    out: Path = src
    if how == "symlink":
        out = tmp_path / "link.json"
        out.symlink_to(src)
    elif how == "relative":
        monkeypatch.chdir(tmp_path)
        out = Path("sub") / ".." / "t.json"
        (tmp_path / "sub").mkdir()
    code, body = run(capsys, src, "--out", out, *force)
    assert code == 2 and body["error"]["type"] == "invalid-argument"
    assert "input transcript" in body["error"]["message"]
    assert src.read_bytes() == before
