"""Tests for JSON output serialization in flyte.cli._common.format."""

import json
from pathlib import Path

from flyte.cli._common import format


def test_json_raw_serializes_pathlib_values():
    # The deploy command renders failed module loads as [("Path", <Path>), ("Err", <str>)].
    # With -o json-raw the values are json.dumps'd; a pathlib.Path must not raise
    # "Object of type PosixPath is not JSON serializable" (FLYTE-SDK-6G).
    vals = [[("Path", Path("/tmp/some/module.py")), ("Err", "boom")]]

    out = format("Modules", vals, of="json-raw")

    parsed = json.loads(out)
    assert parsed == [{"Path": "/tmp/some/module.py", "Err": "boom"}]


def test_json_raw_empty_returns_empty_list():
    assert format("Modules", [], of="json-raw") == "[]"
