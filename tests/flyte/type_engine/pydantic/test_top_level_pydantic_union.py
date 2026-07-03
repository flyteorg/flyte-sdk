"""Regression tests for top-level (non-nested) Pydantic-model unions.

A union whose variants are all plain ``BaseModel`` s resolves every variant to the
same ``"Pydantic Transformer"``. ``UnionTransformer.to_python_value`` previously
relied on the transformer *name* to pick a variant and, failing that, on which
variants ``model_validate``-d without raising. For structurally-overlapping
variants that share a ``type: Literal[...]`` discriminator (e.g. a ``VolumeSource``
union), more than one variant deserialized and it raised a false::

    Ambiguous choice of variant for union type.
    Both Pydantic Transformer and Pydantic Transformer transformers match

The discriminator-aware fast path in ``to_python_value`` resolves these by the
variants' discriminator field (or a unique field-superset) before falling back to
structural matching.
"""

from __future__ import annotations

import typing
from typing import Literal

import msgpack
import pytest
from flyteidl2.core.literals_pb2 import Binary, Literal as _Literal, Scalar, Union
from flyteidl2.core.types_pb2 import LiteralType, SimpleType
from pydantic import BaseModel

from flyte.types import TypeEngine
from flyte.types._type_engine import _resolve_pydantic_union_variant


class LatestForIdentifier(BaseModel):
    model_config = {"frozen": True}
    type: Literal["latest"] = "latest"


class StartFresh(BaseModel):
    model_config = {"frozen": True}
    type: Literal["fresh"] = "fresh"


class PriorRun(BaseModel):
    model_config = {"frozen": True}
    type: Literal["prior_run"] = "prior_run"
    name: str


VolumeSource = typing.Union[LatestForIdentifier, StartFresh, PriorRun]


@pytest.mark.parametrize(
    "value, expected_cls",
    [
        (PriorRun(name="ulcm8fh74pn65kfvzbkb"), PriorRun),
        (LatestForIdentifier(), LatestForIdentifier),
        (StartFresh(), StartFresh),
    ],
)
@pytest.mark.asyncio
async def test_top_level_pydantic_union_roundtrip(value, expected_cls):
    """Each variant survives a full to_literal -> to_python_value round-trip and
    comes back as the exact variant (no false ambiguity)."""
    lt = TypeEngine.to_literal_type(VolumeSource)
    lv = await TypeEngine.to_literal(value, VolumeSource, lt)
    out = await TypeEngine.to_python_value(lv, VolumeSource)
    assert isinstance(out, expected_cls)
    assert out == value


def _untagged_union_literal(value: dict) -> _Literal:
    """Build the literal shape that triggered the original bug: a union scalar
    whose value is a msgpack-encoded pydantic model and whose union ``type`` is a
    bare ``STRUCT`` with **no** ``structure.tag`` (so ``union_tag`` is ``None``).
    This is how a ``--volume '{"type":"prior_run",...}'`` CLI input is stored, and
    what auto-resume reads back from prior runs' inputs."""
    inner = _Literal(scalar=Scalar(binary=Binary(value=msgpack.dumps(value), tag="msgpack")))
    return _Literal(scalar=Scalar(union=Union(value=inner, type=LiteralType(simple=SimpleType.STRUCT))))


@pytest.mark.parametrize(
    "value, expected_cls",
    [
        ({"type": "prior_run", "name": "ulcm8fh74pn65kfvzbkb"}, PriorRun),
        ({"type": "latest"}, LatestForIdentifier),
        ({"type": "fresh"}, StartFresh),
    ],
)
@pytest.mark.asyncio
async def test_untagged_pydantic_union_resolves_by_discriminator(value, expected_cls):
    """The exact regression: an *untagged* union literal previously raised
    'Ambiguous choice of variant ... Both Pydantic Transformer and Pydantic
    Transformer' because multiple default-constructible variants deserialized.
    It must now resolve to the variant named by the discriminator."""
    out = await TypeEngine.to_python_value(_untagged_union_literal(value), VolumeSource)
    assert isinstance(out, expected_cls)


def test_resolve_pydantic_union_variant_by_discriminator():
    variants = [LatestForIdentifier, StartFresh, PriorRun]
    assert _resolve_pydantic_union_variant(variants, {"type": "prior_run", "name": "x"}) is PriorRun
    assert _resolve_pydantic_union_variant(variants, {"type": "latest"}) is LatestForIdentifier
    assert _resolve_pydantic_union_variant(variants, {"type": "fresh"}) is StartFresh


def test_resolve_returns_none_for_single_pydantic_variant():
    # Optional[X] (X + NoneType) must not engage the fast path — falls through unchanged.
    assert _resolve_pydantic_union_variant([PriorRun, type(None)], {"type": "prior_run", "name": "x"}) is None
