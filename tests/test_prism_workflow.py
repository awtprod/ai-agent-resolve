"""Execute the edit-report shell from the workflow against an offline GitHub fixture."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name == "nt" or not shutil.which("bash") or not shutil.which("jq"),
    reason="workflow shell requires POSIX bash and jq",
)

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/prism-review.yml"
HEAD = "a" * 40
BASE = "b" * 40
STALE = "c" * 40
BASE_REF = "dev"


def _target_title(head: str = HEAD, base_ref: str = BASE_REF, base_sha: str = BASE) -> str:
    # Render only the run-name's event expressions so fixtures use the actual workflow identity,
    # including before a regression fix; do not assume a hypothetical base-aware title.
    title = WORKFLOW.read_text().split("run-name: ", 1)[1].splitlines()[0]
    for expression, value in (
        ("github.event.pull_request.number || inputs.pull_request_number", "7"),
        ("github.event.pull_request.head.sha || github.sha", head),
        ("github.event.pull_request.base.ref", base_ref),
        ("github.event.pull_request.base.sha", base_sha),
    ):
        title = title.replace("${{ " + expression + " }}", value)
    assert "${{" not in title
    return title


def _run(
    id_: int = 10,
    *,
    head: str = HEAD,
    base_ref: str = BASE_REF,
    base_sha: str = BASE,
    **fields: object,
) -> dict[str, object]:
    return {
        "id": id_,
        "run_number": id_,
        "event": "pull_request_target",
        "display_title": _target_title(head, base_ref, base_sha),
        "head_sha": base_sha,
        "pull_requests": [],
        "status": "completed",
        "conclusion": "success",
        "html_url": f"https://example.invalid/runs/{id_}",
        **fields,
    }


def _jobs(result: str, at: str = "2026-10-20T12:00:00Z") -> dict[str, object]:
    return {
        "jobs": [
            {"name": "plan", "conclusion": "success", "completed_at": at},
            {"name": "report", "conclusion": result, "completed_at": at},
        ]
    }


def _report(
    tmp_path: Path,
    runs: list[dict[str, object]],
    jobs: dict[str, object],
    **env: str,
) -> subprocess.CompletedProcess[str]:
    # Extract the actual YAML block without a YAML dependency or a second matching algorithm.
    section = WORKFLOW.read_text().split(
        "      - name: Repeat the latest review result after a title or description edit\n", 1
    )[1]
    block = section.split("        run: |\n", 1)[1]
    lines = []
    for line in block.splitlines():
        if line.strip() and not line.startswith("          "):
            break
        lines.append(line)
    script = textwrap.dedent("\n".join(lines))
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"runs": runs, "jobs": jobs}))
    # Only emulate API pagination/query parameters and gh's --jq output option. All run
    # identity, decision selection, and failure handling execute in the workflow's own shell.
    gh = tmp_path / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
            import json
            import os
            import subprocess
            import sys
            from urllib.parse import parse_qs, urlsplit

            args = sys.argv[1:]
            assert args.pop(0) == "api"
            if args[:1] == ["-X"]:
                assert args[1] in ("POST", "PATCH", "DELETE")
                sys.exit(0)
            url = urlsplit(args[0])
            query = parse_qs(url.query)
            with open(os.environ["FIXTURE"]) as stream:
                fixture = json.load(stream)
            if "/actions/workflows/" in url.path:
                runs = fixture["runs"]
                if "head_sha" in query:
                    runs = [r for r in runs if r["head_sha"] == query["head_sha"][0]]
                page = int(query.get("page", ["1"])[0])
                size = int(query.get("per_page", ["100"])[0])
                data = {"workflow_runs": runs[(page - 1) * size:page * size]}
            elif url.path.endswith("/jobs"):
                data = fixture["jobs"][url.path.split("/")[-2]]
            elif "/actions/runs/" in url.path:
                data = {"workflow_id": 42}
            elif url.path.endswith("/comments"):
                data = []
            else:
                raise AssertionError(args)
            raw = json.dumps(data)
            if "--jq" in args:
                result = subprocess.run(
                    ["jq", "-r", args[args.index("--jq") + 1]], input=raw, text=True
                )
                sys.exit(result.returncode)
            print(raw)
            """)
    )
    gh.chmod(0o700)
    return subprocess.run(
        ["bash", "-c", script],
        env={
            "PATH": f"{tmp_path}{os.pathsep}{os.defpath}",
            "FIXTURE": str(fixture),
            "PR_NUMBER": "7",
            "HEAD_SHA": HEAD,
            "BASE_REF": BASE_REF,
            "BASE_SHA": BASE,
            "PR_AUTHOR": "andrew",
            "GITHUB_REPOSITORY": "example/repo",
            "GITHUB_RUN_ID": "9999",
            "GITHUB_RUN_NUMBER": "9999",
            "RUN_URL": "https://example.invalid/runs/9999",
            "PRISM_MIRROR_WAIT_SECONDS": "0",
            **env,
        },
        text=True,
        capture_output=True,
        timeout=20,
    )


def test_target_run_uses_event_identity_even_when_execution_sha_is_base(tmp_path: Path) -> None:
    result = _report(tmp_path, [_run()], {"10": _jobs("success")})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed in https://example.invalid/runs/10" in result.stdout


def test_run_name_and_report_use_immutable_event_base() -> None:
    workflow = WORKFLOW.read_text()
    run_name = workflow.split("run-name: ", 1)[1].splitlines()[0]
    assert "${{ github.event.pull_request.base.ref }}" in run_name
    assert "${{ github.event.pull_request.base.sha }}" in run_name
    report = workflow.split(
        "      - name: Repeat the latest review result after a title or description edit\n", 1
    )[1]
    assert "BASE_REF: ${{ github.event.pull_request.base.ref }}" in report
    assert "BASE_SHA: ${{ github.event.pull_request.base.sha }}" in report


@pytest.mark.parametrize(
    ("old_ref", "old_sha"),
    [("main", STALE), (BASE_REF, STALE), ("main", BASE)],
    ids=["retarget", "same-ref-new-sha", "different-ref-same-sha"],
)
def test_old_base_green_cannot_survive_cancelled_current_base_review(
    tmp_path: Path, old_ref: str, old_sha: str
) -> None:
    result = _report(
        tmp_path,
        [_run(11, conclusion="cancelled"), _run(base_ref=old_ref, base_sha=old_sha)],
        {"10": _jobs("success"), "11": _jobs("cancelled")},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "No completed Prism Review was found" in result.stdout


@pytest.mark.parametrize(
    "env",
    [{"BASE_REF": ""}, {"BASE_REF": "bad\nref"}, {"BASE_SHA": ""}, {"BASE_SHA": "bad"}],
)
def test_missing_or_invalid_base_fails_closed(tmp_path: Path, env: dict[str, str]) -> None:
    result = _report(tmp_path, [_run()], {"10": _jobs("success")}, **env)
    assert result.returncode == 1, result.stdout + result.stderr


def test_stale_head_is_excluded(tmp_path: Path) -> None:
    result = _report(tmp_path, [_run(head=STALE)], {"10": _jobs("success")})
    assert result.returncode == 1, result.stdout + result.stderr
    assert "No completed Prism Review was found" in result.stdout


def test_newer_failure_cannot_fall_back_to_green(tmp_path: Path) -> None:
    result = _report(
        tmp_path,
        [_run(11, conclusion="failure"), _run()],
        {
            "10": _jobs("success"),
            "11": _jobs("failure", "2026-10-20T13:00:00Z"),
        },
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "did not pass in https://example.invalid/runs/11" in result.stdout


@pytest.mark.parametrize("title", ["7", f"7@{HEAD}"])
def test_manual_review_on_exact_pr_head_without_base_is_not_evidence(
    tmp_path: Path, title: str
) -> None:
    result = _report(
        tmp_path,
        [_run(event="workflow_dispatch", head_sha=HEAD, display_title=title)],
        {"10": _jobs("success")},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "No completed Prism Review was found" in result.stdout


def test_manual_review_with_base_looking_title_is_not_immutable_evidence(tmp_path: Path) -> None:
    result = _report(
        tmp_path,
        [_run(event="workflow_dispatch", head_sha=HEAD)],
        {"10": _jobs("success")},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "No completed Prism Review was found" in result.stdout


def test_legacy_target_title_and_mutable_pr_metadata_are_not_evidence(tmp_path: Path) -> None:
    result = _report(
        tmp_path,
        [
            _run(
                display_title=f"7@{HEAD}",
                pull_requests=[
                    {
                        "number": 7,
                        "head": {"sha": HEAD},
                        "base": {"ref": BASE_REF, "sha": BASE},
                    }
                ],
            )
        ],
        {"10": _jobs("success")},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "No completed Prism Review was found" in result.stdout


def test_manual_review_from_main_is_not_evidence(tmp_path: Path) -> None:
    result = _report(
        tmp_path,
        [_run(event="workflow_dispatch", display_title=f"7@{HEAD}")],
        {"10": _jobs("success")},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "No completed Prism Review was found" in result.stdout


def test_history_is_paginated(tmp_path: Path) -> None:
    other_runs = [_run(i, display_title=f"8@{HEAD}") for i in range(100, 200)]
    result = _report(tmp_path, [*other_runs, _run()], {"10": _jobs("success")})
    assert result.returncode == 0, result.stdout + result.stderr


def test_full_history_bound_fails_instead_of_using_older_green(tmp_path: Path) -> None:
    runs = [_run(), *[_run(i, display_title=f"8@{HEAD}") for i in range(100, 1099)]]
    result = _report(tmp_path, runs, {"10": _jobs("success")})
    assert result.returncode == 1, result.stdout + result.stderr
    assert "history exceeded the 1000-run lookup bound" in result.stderr
