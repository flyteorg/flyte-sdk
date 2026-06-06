"""dict -> dataclass coercion when the module uses ``from __future__ import annotations``.

With PEP 563, ``dataclasses.fields(...).type`` is the *string* annotation (e.g. "str"), so the
early type check in DataclassTransformer.assert_type must not compare a real type against that
string. Regression coverage for that path (validation still happens at decode time).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from flyte.types import TypeEngine
from flyte.types._type_engine import TypeTransformerFailedError


@dataclass
class QueryInfoDC:
    query: str  # required, no default
    name: str = "default"
    tags: dict[str, str] | None = None


@pytest.mark.asyncio
async def test_dict_to_dataclass_with_stringized_annotations_full():
    lit = TypeEngine.to_literal_type(QueryInfoDC)
    lv = await TypeEngine.to_literal({"query": "q", "name": "n", "tags": {"a": "b"}}, QueryInfoDC, lit)
    back = await TypeEngine.to_python_value(lv, QueryInfoDC)
    assert back == QueryInfoDC(query="q", name="n", tags={"a": "b"})


@pytest.mark.asyncio
async def test_dict_to_dataclass_with_stringized_annotations_partial():
    lit = TypeEngine.to_literal_type(QueryInfoDC)
    lv = await TypeEngine.to_literal({"query": "q"}, QueryInfoDC, lit)
    back = await TypeEngine.to_python_value(lv, QueryInfoDC)
    assert back == QueryInfoDC(query="q", name="default", tags=None)


@pytest.mark.asyncio
async def test_dict_to_dataclass_with_stringized_annotations_missing_required_raises():
    lit = TypeEngine.to_literal_type(QueryInfoDC)
    with pytest.raises(TypeTransformerFailedError):
        await TypeEngine.to_literal({"name": "n"}, QueryInfoDC, lit)
