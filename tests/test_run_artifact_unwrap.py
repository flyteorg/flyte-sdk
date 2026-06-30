"""
Tests for unwrapping ``flyte.remote.Artifact`` arguments (positional and keyword)
into their underlying ``pb2["data"]`` payloads before a local run is submitted.

See ``flyte._run._unwrap_artifacts`` / ``_unwrap_artifact_value``.
"""

from __future__ import annotations

import pytest

from flyte._run import _unwrap_artifact_value, _unwrap_artifacts
from flyte.remote import Artifact


def _artifact(data):
    """Build an Artifact whose pb2 payload carries ``data`` under the ``data`` key."""
    return Artifact(pb2={"data": data})


# ---------------------------------------------------------------------------
# _unwrap_artifact_value
# ---------------------------------------------------------------------------


class TestUnwrapArtifactValue:
    def test_unwraps_single_artifact(self):
        assert _unwrap_artifact_value(_artifact("hello")) == "hello"

    def test_passes_through_non_artifact(self):
        assert _unwrap_artifact_value(42) == 42
        assert _unwrap_artifact_value("plain") == "plain"
        assert _unwrap_artifact_value(None) is None

    def test_unwraps_artifacts_inside_list(self):
        result = _unwrap_artifact_value([_artifact("a"), _artifact("b")])
        assert result == ["a", "b"]

    def test_list_mixes_artifacts_and_plain_values(self):
        result = _unwrap_artifact_value([_artifact("a"), 1, "x"])
        assert result == ["a", 1, "x"]

    def test_empty_list_returned_unchanged(self):
        value = []
        # Empty lists are not iterated (len == 0) and returned as-is.
        assert _unwrap_artifact_value(value) is value

    def test_dict_value_passed_through(self):
        # Only lists get element-wise treatment; other containers are untouched.
        value = {"k": _artifact("a")}
        assert _unwrap_artifact_value(value) is value


# ---------------------------------------------------------------------------
# _unwrap_artifacts
# ---------------------------------------------------------------------------


class TestUnwrapArtifacts:
    def test_positional_artifacts_are_unwrapped(self):
        new_args, new_kwargs = _unwrap_artifacts((_artifact("x"), 5), {})
        assert new_args == ("x", 5)
        assert new_kwargs == {}

    def test_keyword_artifacts_are_unwrapped(self):
        new_args, new_kwargs = _unwrap_artifacts((), {"a": _artifact("y"), "b": "z"})
        assert new_args == ()
        assert new_kwargs == {"a": "y", "b": "z"}

    def test_mixed_positional_and_keyword(self):
        new_args, new_kwargs = _unwrap_artifacts(
            (_artifact("p0"), "p1"),
            {"k0": _artifact("kv"), "k1": [_artifact("l0"), 2]},
        )
        assert new_args == ("p0", "p1")
        assert new_kwargs == {"k0": "kv", "k1": ["l0", 2]}

    def test_no_artifacts_returns_equivalent_values(self):
        args = (1, "two", [3, 4])
        kwargs = {"a": 1, "b": "two"}
        new_args, new_kwargs = _unwrap_artifacts(args, kwargs)
        assert new_args == args
        assert new_kwargs == kwargs

    def test_empty_args_and_kwargs(self):
        new_args, new_kwargs = _unwrap_artifacts((), {})
        assert new_args == ()
        assert new_kwargs == {}

    def test_returns_new_containers(self):
        args = (1,)
        kwargs = {"a": 1}
        new_args, new_kwargs = _unwrap_artifacts(args, kwargs)
        assert new_args == args
        assert new_kwargs == kwargs
        # A fresh dict is always produced so callers can mutate it safely.
        assert new_kwargs is not kwargs
