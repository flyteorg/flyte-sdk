"""
Tests for unwrapping ``flyte.remote.Artifact`` arguments (positional and keyword)
into the python values stored in their literals before a run is submitted.

See ``flyte._run._unwrap_artifacts`` / ``_unwrap_artifact_value``.
"""

from __future__ import annotations

import pytest
from flyteidl2.artifact import artifact_pb2

from flyte._run import _unwrap_artifact_value, _unwrap_artifacts
from flyte.remote import Artifact
from flyte.types import TypeEngine


async def _artifact(data: str) -> Artifact:
    """Build an Artifact whose spec stores ``data`` as a string literal."""
    lt = TypeEngine.to_literal_type(str)
    lit = await TypeEngine.to_literal(data, str, lt)
    return Artifact(
        pb2=artifact_pb2.Artifact(
            artifact_id=artifact_pb2.ArtifactIdentifier(
                name=artifact_pb2.ArtifactName(org="org", project="proj", domain="dev", name="my_artifact"),
                version="1.0",
            ),
            spec=artifact_pb2.ArtifactSpec(value=lit, type=lt),
        )
    )


# ---------------------------------------------------------------------------
# _unwrap_artifact_value
# ---------------------------------------------------------------------------


class TestUnwrapArtifactValue:
    @pytest.mark.asyncio
    async def test_unwraps_single_artifact(self):
        assert await _unwrap_artifact_value(await _artifact("hello")) == "hello"

    @pytest.mark.asyncio
    async def test_passes_through_non_artifact(self):
        assert await _unwrap_artifact_value(42) == 42
        assert await _unwrap_artifact_value("plain") == "plain"
        assert await _unwrap_artifact_value(None) is None

    @pytest.mark.asyncio
    async def test_unwraps_artifacts_inside_list(self):
        result = await _unwrap_artifact_value([await _artifact("a"), await _artifact("b")])
        assert result == ["a", "b"]

    @pytest.mark.asyncio
    async def test_unwraps_mixed_list(self):
        result = await _unwrap_artifact_value([await _artifact("a"), 1, "x"])
        assert result == ["a", 1, "x"]

    @pytest.mark.asyncio
    async def test_empty_list_passes_through(self):
        value: list = []
        assert await _unwrap_artifact_value(value) is value

    @pytest.mark.asyncio
    async def test_dict_passes_through(self):
        value = {"k": "v"}
        assert await _unwrap_artifact_value(value) is value


# ---------------------------------------------------------------------------
# _unwrap_artifacts
# ---------------------------------------------------------------------------


class TestUnwrapArtifacts:
    @pytest.mark.asyncio
    async def test_positional_artifacts_are_unwrapped(self):
        new_args, new_kwargs = await _unwrap_artifacts((await _artifact("x"), 5), {})
        assert new_args == ("x", 5)
        assert new_kwargs == {}

    @pytest.mark.asyncio
    async def test_keyword_artifacts_are_unwrapped(self):
        new_args, new_kwargs = await _unwrap_artifacts((), {"a": await _artifact("y"), "b": "z"})
        assert new_args == ()
        assert new_kwargs == {"a": "y", "b": "z"}

    @pytest.mark.asyncio
    async def test_mixed_positional_and_keyword(self):
        new_args, new_kwargs = await _unwrap_artifacts(
            (await _artifact("p"),),
            {"k": [await _artifact("l1"), await _artifact("l2")]},
        )
        assert new_args == ("p",)
        assert new_kwargs == {"k": ["l1", "l2"]}

    @pytest.mark.asyncio
    async def test_no_artifacts_returns_equivalent_values(self):
        args = (1, "two", [3])
        kwargs = {"a": None}
        new_args, new_kwargs = await _unwrap_artifacts(args, kwargs)
        assert new_args == args
        assert new_kwargs == kwargs

    @pytest.mark.asyncio
    async def test_empty_args_and_kwargs(self):
        new_args, new_kwargs = await _unwrap_artifacts((), {})
        assert new_args == ()
        assert new_kwargs == {}

    @pytest.mark.asyncio
    async def test_returns_new_containers(self):
        args = (1,)
        kwargs = {"a": 2}
        new_args, new_kwargs = await _unwrap_artifacts(args, kwargs)
        assert new_kwargs is not kwargs
        assert new_args == args
        assert new_kwargs == kwargs
