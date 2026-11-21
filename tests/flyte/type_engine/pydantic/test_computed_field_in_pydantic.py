from typing import Optional, Tuple, Union

import msgpack
import pytest
from pydantic import BaseModel, ConfigDict, computed_field

from flyte.types import TypeEngine


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_days: int = 30

    @computed_field
    @property
    def n_weeks(self) -> int:
        return self.n_days // 7


class Wrapper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: Params
    label: str = "x"

    @computed_field
    @property
    def tag(self) -> str:
        return f"{self.label}-{self.params.n_weeks}"


@pytest.mark.asyncio
async def test_computed_field_with_extra_forbid_roundtrip():
    lt = TypeEngine.to_literal_type(Params)
    lv = await TypeEngine.to_literal(Params(n_days=15), Params, lt)

    # Computed fields stay in the serialized output (so they reach stored results and the UI)...
    assert msgpack.loads(lv.scalar.binary.value) == {"n_days": 15, "n_weeks": 2}

    # ...and are stripped on the way back, so a model with extra="forbid" still round-trips.
    pv = await TypeEngine.to_python_value(lv, Params)
    assert pv == Params(n_days=15)
    assert pv.n_weeks == 2


@pytest.mark.asyncio
async def test_nested_computed_fields_with_extra_forbid_roundtrip():
    value = Wrapper(params=Params(n_days=21), label="run")
    lt = TypeEngine.to_literal_type(Wrapper)
    lv = await TypeEngine.to_literal(value, Wrapper, lt)

    # Computed fields are kept at every level, including the nested model.
    assert msgpack.loads(lv.scalar.binary.value) == {
        "params": {"n_days": 21, "n_weeks": 3},
        "label": "run",
        "tag": "run-3",
    }

    pv = await TypeEngine.to_python_value(lv, Wrapper)
    assert pv == value
    assert pv.tag == "run-3"


@pytest.mark.asyncio
async def test_outputs_stored_before_the_fix_still_load():
    """Outputs serialized by an older SDK already carry computed fields; they must still load."""
    lt = TypeEngine.to_literal_type(Params)
    lv = await TypeEngine.to_literal(Params(n_days=15), Params, lt)

    # Simulate a legacy payload that also included the computed field in the binary.
    lv.scalar.binary.value = msgpack.dumps({"n_days": 15, "n_weeks": 2})

    pv = await TypeEngine.to_python_value(lv, Params)
    assert pv == Params(n_days=15)
    assert pv.n_weeks == 2


@pytest.mark.asyncio
async def test_dict_input_with_computed_field_is_coerced():
    """to_literal coerces a plain dict input; a stray computed-field key must not break extra=forbid."""
    lt = TypeEngine.to_literal_type(Params)
    lv = await TypeEngine.to_literal({"n_days": 14, "n_weeks": 2}, Params, lt)

    pv = await TypeEngine.to_python_value(lv, Params)
    assert pv == Params(n_days=14)
    assert pv.n_weeks == 2


@pytest.mark.asyncio
async def test_namedtuple_holding_a_computed_field_model_roundtrips():
    """A NamedTuple holding a model with computed fields + extra=forbid must round-trip."""
    import typing

    class Pair(typing.NamedTuple):
        params: Params
        name: str

    value = Pair(params=Params(n_days=28), name="job")
    lt = TypeEngine.to_literal_type(Pair)
    lv = await TypeEngine.to_literal(value, Pair, lt)

    pv = await TypeEngine.to_python_value(lv, Pair)
    assert pv.params == Params(n_days=28)
    assert pv.params.n_weeks == 4
    assert pv.name == "job"


@pytest.mark.asyncio
async def test_typeddict_holding_a_computed_field_model_roundtrips():
    """A TypedDict holding a model with computed fields + extra=forbid must round-trip."""
    import typing

    class Payload(typing.TypedDict):
        params: Params
        name: str

    value: Payload = {"params": Params(n_days=35), "name": "job"}
    lt = TypeEngine.to_literal_type(Payload)
    lv = await TypeEngine.to_literal(value, Payload, lt)

    pv = await TypeEngine.to_python_value(lv, Payload)
    assert pv["params"] == Params(n_days=35)
    assert pv["params"].n_weeks == 5
    assert pv["name"] == "job"


class OptionalWrapper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: Optional[Params] = None
    pep604: "Params | None" = None


class UnionWrapper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: Union[Params, int]


class TupleWrapper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair: Tuple[Params, Params]
    rest: Tuple[Params, ...] = ()


@pytest.mark.asyncio
async def test_pep604_optional_nested_computed_field_roundtrips():
    """A `Model | None` field (PEP 604) holding computed fields + extra=forbid must round-trip."""
    value = OptionalWrapper(params=Params(n_days=21), pep604=Params(n_days=28))
    lt = TypeEngine.to_literal_type(OptionalWrapper)
    lv = await TypeEngine.to_literal(value, OptionalWrapper, lt)

    pv = await TypeEngine.to_python_value(lv, OptionalWrapper)
    assert pv.params == Params(n_days=21)
    assert pv.params.n_weeks == 3
    assert pv.pep604 == Params(n_days=28)
    assert pv.pep604.n_weeks == 4


@pytest.mark.asyncio
async def test_union_field_with_computed_field_model_roundtrips():
    """A Union field whose value is a computed-field model + extra=forbid must round-trip."""
    value = UnionWrapper(item=Params(n_days=21))
    lt = TypeEngine.to_literal_type(UnionWrapper)
    lv = await TypeEngine.to_literal(value, UnionWrapper, lt)

    pv = await TypeEngine.to_python_value(lv, UnionWrapper)
    assert pv.item == Params(n_days=21)
    assert pv.item.n_weeks == 3


@pytest.mark.asyncio
async def test_tuple_field_with_computed_field_models_roundtrips():
    """Fixed- and variadic-length tuple fields must recurse into nested computed-field models."""
    value = TupleWrapper(pair=(Params(n_days=7), Params(n_days=21)), rest=(Params(n_days=35),))
    lt = TypeEngine.to_literal_type(TupleWrapper)
    lv = await TypeEngine.to_literal(value, TupleWrapper, lt)

    pv = await TypeEngine.to_python_value(lv, TupleWrapper)
    assert pv.pair[0] == Params(n_days=7)
    assert pv.pair[1].n_weeks == 3
    assert pv.rest[0] == Params(n_days=35)
    assert pv.rest[0].n_weeks == 5


@pytest.mark.asyncio
async def test_wrong_length_fixed_tuple_still_fails_validation():
    """Stripping must not truncate a fixed-length tuple and let a wrong-length payload validate."""
    value = TupleWrapper(pair=(Params(n_days=7), Params(n_days=21)))
    lt = TypeEngine.to_literal_type(TupleWrapper)
    lv = await TypeEngine.to_literal(value, TupleWrapper, lt)
    lv.scalar.binary.value = msgpack.dumps({"pair": [{"n_days": 7}, {"n_days": 14}, {"n_days": 21}], "rest": []})

    with pytest.raises(Exception, match="pair"):
        await TypeEngine.to_python_value(lv, TupleWrapper)
