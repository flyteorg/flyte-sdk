"""
Tests for how ``flyte.remote.Artifact`` arguments reach a run.

Two distinct paths:

- Remote submit coerces the artifact's *stored literal* to the input's declared type by
  round-tripping it through the type engine (``Artifact.coerce_to_literal``), so the engine
  owns every compatibility rule and a mismatch fails at submit time. The coerced literal
  carries the service-stamped ``Literal.artifact_id`` -- provenance is copied, not computed.
  See ``convert.bind_artifact_literals``.
- Local/hybrid runs the task in-process and therefore needs real python values.
  See ``flyte._run._unwrap_artifacts`` / ``_unwrap_artifact_value``.
"""

from __future__ import annotations

from typing import List, Optional

import pytest
from flyteidl2.artifact import artifact_pb2
from flyteidl2.core import artifact_id_pb2

from flyte._internal.runtime.convert import bind_artifact_literals
from flyte._run import _unwrap_artifact_value, _unwrap_artifacts
from flyte.io import Dir, File
from flyte.models import NativeInterface
from flyte.remote import Artifact
from flyte.types import TypeEngine

_VERSION_ID = artifact_id_pb2.ArtifactVersionId(
    key=artifact_id_pb2.ArtifactKey(org="org", project="proj", domain="dev", name="my_artifact"),
    version="1.0",
)


def _wrap(lit, lt) -> Artifact:
    """Wrap a literal as an Artifact, stamping artifact_id the way the service does."""
    lit.artifact_id.CopyFrom(_VERSION_ID)
    return Artifact(
        pb2=artifact_pb2.Artifact(
            artifact_id=artifact_pb2.ArtifactIdentifier(
                name=artifact_pb2.ArtifactName(org="org", project="proj", domain="dev", name="my_artifact"),
                version="1.0",
            ),
            spec=artifact_pb2.ArtifactSpec(value=lit, type=lt),
        )
    )


async def _artifact(data: str) -> Artifact:
    """Build an Artifact whose spec stores ``data`` as a string literal."""
    lt = TypeEngine.to_literal_type(str)
    return _wrap(await TypeEngine.to_literal(data, str, lt), lt)


async def _artifact_of_type(python_type, uri: str = "s3://bucket/obj") -> Artifact:
    """Build an Artifact storing an offloaded asset of ``python_type`` (File/Dir)."""
    lt = TypeEngine.to_literal_type(python_type)
    return _wrap(await TypeEngine.to_literal(python_type(path=uri), python_type, lt), lt)


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
# Direct literal binding (remote submit path)
# ---------------------------------------------------------------------------


def _interface(fn) -> NativeInterface:
    return NativeInterface.from_callable(fn)


class TestBindArtifactLiterals:
    @pytest.mark.asyncio
    async def test_binds_stored_value(self):
        """The bound literal is the stored value coerced to the declared type -- same content."""
        art = await _artifact("hello")

        def task(v: str): ...

        bound, remaining = await bind_artifact_literals(_interface(task), (), {"v": art})

        assert remaining == {}
        assert bound["v"].scalar.primitive.string_value == "hello"

    @pytest.mark.asyncio
    async def test_bound_literal_keeps_service_artifact_id(self):
        """Provenance is copied from the service's stamp onto the coerced literal."""
        art = await _artifact("hello")

        def task(v: str): ...

        bound, _ = await bind_artifact_literals(_interface(task), (), {"v": art})
        assert bound["v"].artifact_id == _VERSION_ID

    @pytest.mark.asyncio
    async def test_positional_artifact_is_named(self):
        art = await _artifact("hello")

        def task(v: str, w: int): ...

        bound, remaining = await bind_artifact_literals(_interface(task), (art, 5), {})
        assert set(bound) == {"v"}
        assert remaining == {"w": 5}

    @pytest.mark.asyncio
    async def test_list_of_artifacts_becomes_collection(self):
        arts = [await _artifact("a"), await _artifact("b")]

        def task(v: List[str]): ...

        bound, _ = await bind_artifact_literals(_interface(task), (), {"v": arts})

        elements = bound["v"].collection.literals
        assert [e.scalar.primitive.string_value for e in elements] == ["a", "b"]
        assert all(e.artifact_id == _VERSION_ID for e in elements)

    @pytest.mark.asyncio
    async def test_mixed_list_converts_plain_elements(self):
        def task(v: List[str]): ...

        bound, _ = await bind_artifact_literals(_interface(task), (), {"v": [await _artifact("a"), "plain"]})

        elements = bound["v"].collection.literals
        assert [e.scalar.primitive.string_value for e in elements] == ["a", "plain"]
        assert elements[0].artifact_id == _VERSION_ID
        assert not elements[1].HasField("artifact_id")

    @pytest.mark.asyncio
    async def test_non_artifact_args_pass_through(self):
        def task(v: str, w: int): ...

        bound, remaining = await bind_artifact_literals(_interface(task), (), {"v": "x", "w": 1})
        assert bound == {}
        assert remaining == {"v": "x", "w": 1}


class TestBindArtifactTypeChecking:
    @pytest.mark.asyncio
    async def test_type_mismatch_raises_at_submit(self):
        """A File artifact bound to a `str` input used to blow up inside the task."""
        art = await _artifact_of_type(File)

        def task(v: str): ...

        with pytest.raises(ValueError, match="cannot bind to input 'v'"):
            await bind_artifact_literals(_interface(task), (), {"v": art})

    @pytest.mark.asyncio
    async def test_file_and_dir_are_not_interchangeable(self):
        """Both are blobs; only `dimensionality` separates them."""
        art = await _artifact_of_type(File)

        def task(v: Dir): ...

        with pytest.raises(ValueError, match="cannot bind to input 'v'"):
            await bind_artifact_literals(_interface(task), (), {"v": art})

    @pytest.mark.asyncio
    async def test_matching_type_binds(self):
        art = await _artifact_of_type(File)

        def task(v: File): ...

        bound, _ = await bind_artifact_literals(_interface(task), (), {"v": art})
        assert bound["v"].scalar.blob.uri == art.pb2.spec.value.scalar.blob.uri
        assert bound["v"].artifact_id == _VERSION_ID

    @pytest.mark.asyncio
    async def test_optional_input_wraps_in_union(self):
        """Coercion produces the runtime shape the task expects: Optional[File] is a union
        literal, not a bare blob -- something verbatim binding used to get wrong."""
        art = await _artifact_of_type(File)

        def task(v: Optional[File] = None): ...

        bound, _ = await bind_artifact_literals(_interface(task), (), {"v": art})
        assert bound["v"].scalar.WhichOneof("value") == "union"
        assert bound["v"].scalar.union.value.scalar.blob.uri == art.pb2.spec.value.scalar.blob.uri
        assert bound["v"].artifact_id == _VERSION_ID

    @pytest.mark.asyncio
    async def test_list_element_type_is_checked(self):
        def task(v: List[File]): ...

        with pytest.raises(ValueError, match=r"cannot bind to input 'v\[0\]'"):
            await bind_artifact_literals(_interface(task), (), {"v": [await _artifact("a")]})


class TestNestedArtifactRejected:
    @pytest.mark.asyncio
    async def test_artifact_in_dict_raises(self):
        def task(cfg: dict): ...

        with pytest.raises(ValueError, match="argument 'cfg' has an Artifact nested inside a container"):
            await bind_artifact_literals(_interface(task), (), {"cfg": {"model": await _artifact("a")}})

    @pytest.mark.asyncio
    async def test_artifact_in_nested_list_raises(self):
        def task(v: list): ...

        with pytest.raises(ValueError, match="argument 'v' has an Artifact nested inside a container"):
            await bind_artifact_literals(_interface(task), (), {"v": [[await _artifact("a")]]})

    @pytest.mark.asyncio
    async def test_plain_dict_passes_through(self):
        def task(cfg: dict): ...

        bound, remaining = await bind_artifact_literals(_interface(task), (), {"cfg": {"k": "v"}})
        assert bound == {}
        assert remaining == {"cfg": {"k": "v"}}
