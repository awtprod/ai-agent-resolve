"""Static pins and input validation for ``aar.deliver``.

- The package never runs a shell, never imports a scratch-dir module, and holds no shared-temp
  literal (so delivery work stays in the caller's durable work dir, inside pytest ``tmp_path``).
- ``RenderOutput.from_measured`` refuses missing or ill-typed keys.
- Every stable error code is documented in DESIGN.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import aar.deliver
from aar.deliver.errors import ERROR_CODES
from aar.deliver.model import RenderOutput

_PACKAGE = Path(aar.deliver.__file__).parent
_DESIGN = Path(__file__).resolve().parents[2] / "docs" / "DESIGN.md"
_FORBIDDEN_MODULES = {"tempfile"}
_FORBIDDEN_LITERAL = "/tmp"


def _sources() -> list[tuple[str, ast.AST]]:
    return [(p.name, ast.parse(p.read_text(encoding="utf-8"))) for p in _PACKAGE.glob("*.py")]


def test_no_shell_true_no_tempfile_no_temp_literal() -> None:
    for name, tree in _sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in _FORBIDDEN_MODULES, name
            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root not in _FORBIDDEN_MODULES, name
            if isinstance(node, ast.keyword) and node.arg == "shell":
                assert not (isinstance(node.value, ast.Constant) and node.value.value is True), (
                    f"{name}: shell=True is forbidden"
                )
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert _FORBIDDEN_LITERAL not in node.value, (
                    f"{name}: holds a {_FORBIDDEN_LITERAL} literal"
                )


# -- RenderOutput.from_measured --------------------------------------------------------------

_GOOD = {
    "output_path": "D:/Out/show_v1.mov",
    "frame_count": 100,
    "frame_rate": 29.97,
    "duration_seconds": 3.4,
    "render_job_id": "rj-1",
}


def test_from_measured_accepts_a_good_dict() -> None:
    out = RenderOutput.from_measured(dict(_GOOD))
    assert out.output_path == "D:/Out/show_v1.mov"
    assert out.frame_count == 100
    assert out.render_job_id == "rj-1"


@pytest.mark.parametrize(
    "mutate",
    [
        {"output_path": None},  # missing
        {"frame_rate": "fast"},  # ill-typed
        {"frame_count": True},  # bool is not a frame count
        {"render_job_id": ""},  # empty id
        {"duration_seconds": 0},  # non-positive
        {"frame_rate": -1.0},  # non-positive
    ],
)
def test_from_measured_refuses_bad_keys(mutate: dict) -> None:
    from aar.deliver.errors import DeliverError

    data = {**_GOOD, **mutate}
    with pytest.raises(DeliverError) as exc:
        RenderOutput.from_measured(data)
    assert exc.value.code == "invalid_render_output"


def test_from_measured_refuses_a_non_dict() -> None:
    from aar.deliver.errors import DeliverError

    with pytest.raises(DeliverError):
        RenderOutput.from_measured(["not", "a", "dict"])


# -- documented error codes ------------------------------------------------------------------


def test_every_error_code_is_documented() -> None:
    text = _DESIGN.read_text(encoding="utf-8")
    missing = [code for code in ERROR_CODES if code not in text]
    assert missing == [], f"undocumented DeliverError codes: {missing}"
