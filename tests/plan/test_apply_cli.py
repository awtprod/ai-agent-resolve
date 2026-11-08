"""``aar plan apply`` end to end: golden output, lint refusal, --submit guards, idempotency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from apply_builders import plan_data, seg
from plan_support import FIXTURES

from aar.cli import main

LONGFORM = str(FIXTURES / "ccn_longform_v2.plan.json")
SHORT = str(FIXTURES / "short_v2.plan.json")
TEMPLATE = str(FIXTURES / "ccn_longform_v2.template.json")
PREPARE = str(FIXTURES / "ccn_longform_v2.prepare.json")
ASSEMBLE = str(FIXTURES / "ccn_longform_v2.assemble.json")
BRANCH = str(FIXTURES / "ccn_longform_v2.branch.json")
LF_INPUTS = [
    "--template-describe",
    TEMPLATE,
    "--prepare-result",
    PREPARE,
    "--assemble-result",
    ASSEMBLE,
]
SHORT_INPUTS = [
    "--template-describe",
    str(FIXTURES / "short_v2.template.json"),
    "--prepare-result",
    str(FIXTURES / "short_v2.prepare.json"),
    "--assemble-result",
    str(FIXTURES / "short_v2.assemble.json"),
]


def _golden(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_longform_golden(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "apply", LONGFORM, *LF_INPUTS]) == 0
    assert capsys.readouterr().out == _golden("ccn_longform_v2.apply.txt")


def test_short_golden(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "apply", SHORT, *SHORT_INPUTS]) == 0
    assert capsys.readouterr().out == _golden("short_v2.apply.txt")


def test_longform_json_golden(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "apply", LONGFORM, *LF_INPUTS, "--json"]) == 0
    assert capsys.readouterr().out == _golden("ccn_longform_v2.apply.json")


@pytest.mark.parametrize("extra", [[], ["--json"]])
def test_lint_refusal_exits_2(capsys: pytest.CaptureFixture[str], extra: list[str]) -> None:
    broken = str(FIXTURES / "short_broken.plan.json")
    assert main(["plan", "apply", broken, *extra]) == 2
    cap = capsys.readouterr()
    if extra:
        assert json.loads(cap.out)["ok"] is False
    else:
        assert "lint_errors" in cap.err


@pytest.mark.parametrize("phase", ["prepare", "assemble", "finish"])
def test_submit_without_inputs_refused(capsys: pytest.CaptureFixture[str], phase: str) -> None:
    assert main(["plan", "apply", LONGFORM, "--submit", phase]) == 2
    assert "unresolved_reference" in capsys.readouterr().err


def test_submit_assemble_needs_prepare_result(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["plan", "apply", LONGFORM, "--template-describe", TEMPLATE, "--submit", "assemble"]
    )
    assert code == 2
    assert "unresolved_reference" in capsys.readouterr().err


def test_submit_finish_needs_results(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["plan", "apply", LONGFORM, "--prepare-result", PREPARE, "--submit", "finish"])
    assert code == 2
    assert "unresolved_reference" in capsys.readouterr().err


def _walk(obj: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found.extend(_walk(v, key))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_walk(v, key))
    return found


@pytest.mark.parametrize("phase", ["prepare", "assemble", "finish"])
def test_submit_with_all_inputs(capsys: pytest.CaptureFixture[str], phase: str) -> None:
    assert main(["plan", "apply", LONGFORM, *LF_INPUTS, "--submit", phase, "--json"]) == 0
    out = capsys.readouterr().out
    body = json.loads(out)
    assert body["kind"] == "ops.sequence"
    assert body["phase"] == phase
    assert body["idempotency_key"].startswith("plan.apply:")
    assert body["params"]["steps"]
    assert "__unresolved__" not in out
    plan_paths = {s["media"] for s in json.loads(Path(LONGFORM).read_text())["sources"]}
    uids = _walk(body["params"], "unique_id")
    assert not plan_paths & {u for u in uids if isinstance(u, str)}
    for step in body["params"]["steps"]:
        assert not plan_paths & {
            u for u in _walk(step["params"], "unique_id") if isinstance(u, str)
        }


def test_submit_text_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "apply", LONGFORM, *LF_INPUTS, "--submit", "finish"]) == 0
    out = capsys.readouterr().out
    assert "idempotency-key: plan.apply:" in out


def test_idempotency_diff_lists_only_missing(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "apply", LONGFORM, *LF_INPUTS, "--branch-describe", BRANCH]) == 0
    out = capsys.readouterr().out
    assert "only what is missing" in out
    assert "appends: 2 present, 4 to add" in out
    assert "(segment 2)" in out and "(segment 3)" in out
    assert "(segment 0)" not in out.split("IDEMPOTENCY")[1]
    assert "markers: 1 present, 1 to add" in out


def test_template_describe_wrong_name_refused(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["plan", "apply", SHORT, "--template-describe", TEMPLATE]  # longform template, short plan
    )
    assert code == 2
    assert "template_name_mismatch" in capsys.readouterr().err


def _write(tmp_path: Path, data: dict[str, Any]) -> str:
    path = tmp_path / "hostile.plan.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_hostile_gain_exits_2_typed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path, plan_data(segments=[seg(gain_db=45.0)]))
    assert main(["plan", "apply", path]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_hostile_title_track_17_exits_2_typed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    segs = [seg(track=t, a=t * 100, b=t * 100 + 24) for t in range(1, 17)]
    data = plan_data(
        segments=segs,
        titles=[{"lines": ["Hi"], "record_in": 0, "record_out": 10, "template": "T"}],
    )
    path = _write(tmp_path, data)
    assert main(["plan", "apply", path]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_w43_finish_submit_key_depends_on_the_assemble_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Two assemble results that differ (by job_id bytes) give different finish keys, so a re-run
    # after an assemble redo is a new key, never a conflict (F2 at the CLI; kills W43).
    import copy
    import json as _json

    base = _json.loads((FIXTURES / "ccn_longform_v2.assemble.json").read_text())
    a1 = tmp_path / "asm1.json"
    a1.write_text(_json.dumps(base))
    other = copy.deepcopy(base)
    other["job_id"] = "job_assemble_REDO"
    a2 = tmp_path / "asm2.json"
    a2.write_text(_json.dumps(other))

    def key(asm: str) -> str:
        assert (
            main(
                [
                    "plan",
                    "apply",
                    LONGFORM,
                    "--submit",
                    "finish",
                    "--json",
                    "--prepare-result",
                    PREPARE,
                    "--assemble-result",
                    asm,
                ]
            )
            == 0
        )
        return _json.loads(capsys.readouterr().out)["idempotency_key"]

    assert key(str(a1)) != key(str(a2))


def test_submit_prints_warnings_before_submitting(capsys: pytest.CaptureFixture[str]) -> None:
    # The mute warnings (and other notes) are surfaced on the submit path, not hidden (R4).
    code = main(
        ["plan", "apply", LONGFORM, "--submit", "assemble", "--json", "--prepare-result", PREPARE]
    )
    assert code == 0
    assert "mute is template state" in capsys.readouterr().err
