"""
Tests for unwrapping ``flyte.remote.Artifact`` arguments (positional and keyword)
into the python values stored in their literals before a run is submitted, and for
stamping artifact provenance onto converted run inputs.

See ``flyte._run._unwrap_artifacts`` / ``_unwrap_artifact_value`` / ``_stamp_artifact_inputs``.
"""

from __future__ import annotations

from typing import List

import pytest
from flyteidl2.artifact import artifact_pb2
from flyteidl2.task import common_pb2

from flyteidl2.core import artifact_id_pb2

from flyte._internal.runtime.convert import Inputs
from flyte._run import _stamp_artifact_inputs, _unwrap_artifact_value, _unwrap_artifacts
from flyte.remote import Artifact
from flyte.types import TypeEngine

_VERSION_ID = artifact_id_pb2.ArtifactVersionId(
    key=artifact_id_pb2.ArtifactKey(org="org", project="proj", domain="dev", name="my_artifact"),
    version="1.0",
)


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
    async def test_unwraps_single_artifact_with_tracker(self):
        value, source = await _unwrap_artifact_value(await _artifact("hello"))
        assert value == "hello"
        assert source == _VERSION_ID

    @pytest.mark.asyncio
    async def test_passes_through_non_artifact(self):
        assert await _unwrap_artifact_value(42) == (42, None)
        assert await _unwrap_artifact_value("plain") == ("plain", None)
        assert await _unwrap_artifact_value(None) == (None, None)

    @pytest.mark.asyncio
    async def test_unwraps_artifacts_inside_list(self):
        value, source = await _unwrap_artifact_value([await _artifact("a"), await _artifact("b")])
        assert value == ["a", "b"]
        assert source == [(0, _VERSION_ID), (1, _VERSION_ID)]

    @pytest.mark.asyncio
    async def test_unwraps_mixed_list(self):
        value, source = await _unwrap_artifact_value([await _artifact("a"), 1, "x"])
        assert value == ["a", 1, "x"]
        assert source == [(0, _VERSION_ID)]

    @pytest.mark.asyncio
    async def test_plain_list_passes_through(self):
        value = [1, 2]
        unwrapped, source = await _unwrap_artifact_value(value)
        assert unwrapped is value
        assert source is None

    @pytest.mark.asyncio
    async def test_dict_passes_through(self):
        value = {"k": "v"}
        unwrapped, source = await _unwrap_artifact_value(value)
        assert unwrapped is value
        assert source is None


# ---------------------------------------------------------------------------
# _unwrap_artifacts
# ---------------------------------------------------------------------------


class TestUnwrapArtifacts:
    @pytest.mark.asyncio
    async def test_positional_artifacts_are_unwrapped(self):
        new_args, new_kwargs, sources = await _unwrap_artifacts((await _artifact("x"), 5), {})
        assert new_args == ("x", 5)
        assert new_kwargs == {}
        assert sources == {0: _VERSION_ID}

    @pytest.mark.asyncio
    async def test_keyword_artifacts_are_unwrapped(self):
        new_args, new_kwargs, sources = await _unwrap_artifacts((), {"a": await _artifact("y"), "b": "z"})
        assert new_args == ()
        assert new_kwargs == {"a": "y", "b": "z"}
        assert sources == {"a": _VERSION_ID}

    @pytest.mark.asyncio
    async def test_mixed_positional_and_keyword(self):
        new_args, new_kwargs, sources = await _unwrap_artifacts(
            (await _artifact("p"),),
            {"k": [await _artifact("l1"), await _artifact("l2")]},
        )
        assert new_args == ("p",)
        assert new_kwargs == {"k": ["l1", "l2"]}
        assert sources == {0: _VERSION_ID, "k": [(0, _VERSION_ID), (1, _VERSION_ID)]}

    @pytest.mark.asyncio
    async def test_no_artifacts_returns_equivalent_values(self):
        args = (1, "two", [3])
        kwargs = {"a": None}
        new_args, new_kwargs, sources = await _unwrap_artifacts(args, kwargs)
        assert new_args == args
        assert new_kwargs == kwargs
        assert sources == {}

    @pytest.mark.asyncio
    async def test_empty_args_and_kwargs(self):
        new_args, new_kwargs, sources = await _unwrap_artifacts((), {})
        assert new_args == ()
        assert new_kwargs == {}
        assert sources == {}


# ---------------------------------------------------------------------------
# _stamp_artifact_inputs
# ---------------------------------------------------------------------------


async def _converted_inputs(**values) -> Inputs:
    """Build converted run inputs. Each value is (python_value, python_type)."""
    literals = []
    for name, (value, python_type) in values.items():
        lt = TypeEngine.to_literal_type(python_type)
        lit = await TypeEngine.to_literal(value, python_type, lt)
        literals.append(common_pb2.NamedLiteral(name=name, value=lit))
    return Inputs(proto_inputs=common_pb2.Inputs(literals=literals))


class TestStampArtifactInputs:
    @pytest.mark.asyncio
    async def test_stamps_positional_and_keyword_inputs(self):
        inputs = await _converted_inputs(v=("hello", str), w=(5, int))
        _stamp_artifact_inputs(inputs, ["v", "w"], {0: _VERSION_ID})

        by_name = {nl.name: nl.value for nl in inputs.proto_inputs.literals}
        assert by_name["v"].artifact_id == _VERSION_ID
        assert not by_name["w"].HasField("artifact_id")
        # No metadata contract keys anywhere.
        assert not by_name["v"].metadata

    @pytest.mark.asyncio
    async def test_stamps_list_elements(self):
        inputs = await _converted_inputs(v=(["a", "b", "c"], List[str]))
        _stamp_artifact_inputs(inputs, ["v"], {"v": [(0, _VERSION_ID), (2, _VERSION_ID)]})

        elements = inputs.proto_inputs.literals[0].value.collection.literals
        assert elements[0].artifact_id == _VERSION_ID
        assert not elements[1].HasField("artifact_id")
        assert elements[2].artifact_id == _VERSION_ID

    @pytest.mark.asyncio
    async def test_identity_surfaces_in_string_repr(self):
        from flyte.types import literal_string_repr

        inputs = await _converted_inputs(v=("hello", str))
        _stamp_artifact_inputs(inputs, ["v"], {"v": _VERSION_ID})
        assert literal_string_repr(inputs.proto_inputs) == {"v": "hello (artifact: org/proj/dev/my_artifact@1.0)"}
