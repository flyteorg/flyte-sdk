"""Tests for ListConfigTransformer."""

from __future__ import annotations

import flyte
import msgpack
import pytest
from flyte.types._type_engine import MESSAGEPACK, TypeEngine
from flyteidl2.core.literals_pb2 import Binary, Literal, Scalar
from omegaconf import ListConfig, OmegaConf

from flyteplugins.omegaconf.base_transformer import ListConfigTransformer
from flyteplugins.omegaconf.codec import (
    KIND_LIST,
    PAYLOAD_KIND,
    PAYLOAD_MARKER,
    PAYLOAD_VALUES,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_transformer() -> ListConfigTransformer:
    return ListConfigTransformer()


async def roundtrip(cfg: ListConfig) -> ListConfig:
    t = make_transformer()
    lt = t.get_literal_type(ListConfig)
    literal = await t.to_literal(cfg, ListConfig, lt)
    return await t.to_python_value(literal, ListConfig)


# ── Registration ──────────────────────────────────────────────────────────────


def test_transformer_registered():
    transformer = TypeEngine.get_transformer(ListConfig)
    assert isinstance(transformer, ListConfigTransformer)


def test_get_literal_type_is_struct():
    from flyteidl2.core.types_pb2 import SimpleType

    t = make_transformer()
    lt = t.get_literal_type(ListConfig)
    assert lt.simple == SimpleType.STRUCT


def test_get_literal_type_has_msgpack_annotation():
    from flyte.types._type_engine import CACHE_KEY_METADATA, SERIALIZATION_FORMAT

    t = make_transformer()
    lt = t.get_literal_type(ListConfig)
    annotations = dict(lt.annotation.annotations)
    assert CACHE_KEY_METADATA in annotations
    assert annotations[CACHE_KEY_METADATA][SERIALIZATION_FORMAT] == MESSAGEPACK


# ── Simple list roundtrips ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_simple_integers():
    cfg = OmegaConf.create([1, 2, 3])
    result = await roundtrip(cfg)
    assert isinstance(result, ListConfig)
    assert list(result) == [1, 2, 3]


@pytest.mark.asyncio
async def test_simple_floats():
    cfg = OmegaConf.create([0.001, 0.01, 0.1])
    result = await roundtrip(cfg)
    assert list(result) == pytest.approx([0.001, 0.01, 0.1])


@pytest.mark.asyncio
async def test_simple_strings():
    cfg = OmegaConf.create(["adam", "sgd", "rmsprop"])
    result = await roundtrip(cfg)
    assert list(result) == ["adam", "sgd", "rmsprop"]


@pytest.mark.asyncio
async def test_simple_booleans():
    cfg = OmegaConf.create([True, False, True])
    result = await roundtrip(cfg)
    assert list(result) == [True, False, True]


@pytest.mark.asyncio
async def test_empty_list():
    cfg = OmegaConf.create([])
    result = await roundtrip(cfg)
    assert isinstance(result, ListConfig)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_single_element():
    cfg = OmegaConf.create([42])
    result = await roundtrip(cfg)
    assert list(result) == [42]


# ── Mixed-type lists ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mixed_types():
    cfg = OmegaConf.create([1, "two", 3.0, True, None])
    result = await roundtrip(cfg)
    assert result[0] == 1
    assert result[1] == "two"
    assert result[2] == pytest.approx(3.0)
    assert result[3] is True
    assert result[4] is None


# ── Nested lists ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nested_list_of_lists():
    cfg = OmegaConf.create([[1, 2], [3, 4], [5, 6]])
    result = await roundtrip(cfg)
    assert isinstance(result, ListConfig)
    assert list(result[0]) == [1, 2]
    assert list(result[1]) == [3, 4]
    assert list(result[2]) == [5, 6]


@pytest.mark.asyncio
async def test_deeply_nested_list():
    cfg = OmegaConf.create([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    result = await roundtrip(cfg)
    assert list(result[0][0]) == [1, 2]
    assert list(result[1][1]) == [7, 8]


@pytest.mark.asyncio
async def test_grid_of_floats():
    """Typical hyperparameter grid: list of LR schedules."""
    cfg = OmegaConf.create(
        [
            [0.001, 0.01, 0.1],
            [10, 20, 50],
        ]
    )
    result = await roundtrip(cfg)
    assert list(result[0]) == pytest.approx([0.001, 0.01, 0.1])
    assert list(result[1]) == [10, 20, 50]


# ── List of dicts ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_of_flat_dicts():
    cfg = OmegaConf.create([{"a": 1}, {"b": 2}, {"c": 3}])
    result = await roundtrip(cfg)
    assert isinstance(result, ListConfig)
    assert result[0].a == 1
    assert result[1].b == 2
    assert result[2].c == 3


@pytest.mark.asyncio
async def test_list_of_nested_dicts():
    cfg = OmegaConf.create(
        [
            {"optimizer": {"lr": 0.001}, "training": {"epochs": 10}},
            {"optimizer": {"lr": 0.01}, "training": {"epochs": 20}},
            {"optimizer": {"lr": 0.1}, "training": {"epochs": 5}},
        ]
    )
    result = await roundtrip(cfg)
    assert result[0].optimizer.lr == pytest.approx(0.001)
    assert result[1].training.epochs == 20
    assert result[2].optimizer.lr == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_list_of_dicts_with_list_values():
    """Each DictConfig element itself has list values."""
    cfg = OmegaConf.create(
        [
            {"name": "encoder", "layer_sizes": [768, 512, 256]},
            {"name": "bottleneck", "layer_sizes": [128]},
            {"name": "decoder", "layer_sizes": [256, 512, 768]},
        ]
    )
    result = await roundtrip(cfg)
    assert result[0].name == "encoder"
    assert list(result[0].layer_sizes) == [768, 512, 256]
    assert result[1].name == "bottleneck"
    assert list(result[1].layer_sizes) == [128]
    assert list(result[2].layer_sizes) == [256, 512, 768]


# ── Interpolation ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_interpolation_resolved():
    """OmegaConf list interpolation is resolved at serialization."""
    cfg = OmegaConf.create([1.0, "${0}"])  # second element references first
    result = await roundtrip(cfg)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(1.0)


# ── LR schedule patterns ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lr_schedule():
    """Realistic use case: LR schedule as ListConfig."""
    base_lr = 0.1
    schedule = OmegaConf.create([base_lr * (0.5**i) for i in range(6)])
    result = await roundtrip(schedule)
    assert len(result) == 6
    assert result[0] == pytest.approx(0.1)
    assert result[1] == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_activation_names():
    """List of strings: activation function names."""
    cfg = OmegaConf.create(["relu", "relu", "sigmoid"])
    result = await roundtrip(cfg)
    assert list(result) == ["relu", "relu", "sigmoid"]


# ── Wire format ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_literal_is_binary_msgpack():
    t = make_transformer()
    cfg = OmegaConf.create([1, 2, 3])
    lt = t.get_literal_type(ListConfig)
    literal = await t.to_literal(cfg, ListConfig, lt)
    assert literal.HasField("scalar")
    assert literal.scalar.HasField("binary")
    assert literal.scalar.binary.tag == MESSAGEPACK


@pytest.mark.asyncio
async def test_payload_is_plain_list():
    """The msgpack payload is a marked ListConfig payload."""
    t = make_transformer()
    cfg = OmegaConf.create([10, 20, 30])
    lt = t.get_literal_type(ListConfig)
    literal = await t.to_literal(cfg, ListConfig, lt)
    payload = msgpack.loads(literal.scalar.binary.value, strict_map_key=False)
    assert payload[PAYLOAD_MARKER] is True
    assert payload[PAYLOAD_KIND] == KIND_LIST
    assert payload[PAYLOAD_VALUES] == [10, 20, 30]


def test_from_binary_idl_wrong_tag_raises():
    t = make_transformer()
    binary = Binary(value=b"data", tag="json")
    with pytest.raises(TypeError, match="Unsupported binary format"):
        t.from_binary_idl(binary, ListConfig)


@pytest.mark.asyncio
async def test_to_python_value_non_binary_raises():
    t = make_transformer()
    literal = Literal(scalar=Scalar())
    with pytest.raises(TypeError):
        await t.to_python_value(literal, ListConfig)


def test_from_binary_idl_direct():
    """from_binary_idl can be called directly with a valid Binary."""
    t = make_transformer()
    payload = {
        PAYLOAD_MARKER: True,
        PAYLOAD_KIND: KIND_LIST,
        PAYLOAD_VALUES: [1, 2, 3],
    }
    binary = Binary(value=msgpack.dumps(payload), tag=MESSAGEPACK)
    result = t.from_binary_idl(binary, ListConfig)
    assert isinstance(result, ListConfig)
    assert list(result) == [1, 2, 3]


# ── Task definitions ──────────────────────────────────────────────────────────

_env = flyte.TaskEnvironment(name="omegaconf-listconfig-test")


@_env.task
async def _identity(cfg: ListConfig) -> ListConfig:
    return cfg


@_env.task
async def _append(cfg: ListConfig, value: float) -> ListConfig:
    return OmegaConf.create([*list(cfg), value])


# ── Task-based integration tests ──────────────────────────────────────────────


def test_task_simple_listconfig_roundtrip():
    flyte.init()
    cfg = OmegaConf.create([1.0, 0.5, 0.25, 0.125])
    run = flyte.run(_identity, cfg=cfg)
    result = run.outputs()[0]
    assert isinstance(result, ListConfig)
    assert list(result) == pytest.approx([1.0, 0.5, 0.25, 0.125])


def test_task_listconfig_of_dicts_roundtrip():
    flyte.init()
    cfg = OmegaConf.create(
        [
            {"name": "layer1", "size": 256},
            {"name": "layer2", "size": 128},
        ]
    )
    run = flyte.run(_identity, cfg=cfg)
    result = run.outputs()[0]
    assert result[0].name == "layer1"
    assert result[1].size == 128


def test_task_append_to_listconfig():
    flyte.init()
    cfg = OmegaConf.create([0.1, 0.01])
    run = flyte.run(_append, cfg=cfg, value=0.001)
    result = run.outputs()[0]
    assert list(result) == pytest.approx([0.1, 0.01, 0.001])
