"""``aar node ops``: the op listing, as text and JSON."""

from __future__ import annotations

import json

import pytest

from aar.cli import main
from aar.node.ops import list_ops


def test_node_ops_json_lists_every_op_with_flags_and_params_schema(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["node", "ops", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [d["name"] for d in listed] == [s.name for s in list_ops()]
    by_name = {d["name"]: d for d in listed}
    append = by_name["timeline.append_ranges"]
    assert (append["hot"], append["mutates"]) == (True, True)
    assert "append-fps-short" in append["traps"]
    assert {"ranges", "cursor", "max_ranges"} <= set(append["params_schema"]["properties"])
    assert append["params_schema"]["required"] == ["ranges"]
    assert (by_name["timeline.describe"]["hot"], by_name["timeline.describe"]["mutates"]) == (
        True,
        False,
    )


def test_node_ops_text_shows_one_entry_per_op(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["node", "ops"]) == 0
    out = capsys.readouterr().out
    for spec in list_ops():
        assert spec.name in out
    assert "hot mutates" in out and "hot read" in out
    assert "params: timeline?, ranges, cursor?" in out


def test_node_without_a_command_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["node"]) == 2
    assert "COMMAND" in capsys.readouterr().err


def test_the_published_timeline_ref_text_states_the_id_rule(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Agents read this schema text: it must not say a name is enough for a write.
    assert main(["node", "ops", "--json"]) == 0
    listed = {d["name"]: d for d in json.loads(capsys.readouterr().out)}
    text = listed["markers.add"]["params_schema"]["$defs"]["TimelineRef"]["description"]
    assert "unique_id" in text and "timeline_id_required" in text
    assert "timeline_ref_mismatch" in text
    assert "or name" not in text and "preferred" not in text
