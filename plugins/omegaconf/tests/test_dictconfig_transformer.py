"""Tests for DictConfigTransformer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import flyte
import msgpack
import pytest
from flyte.types._type_engine import MESSAGEPACK, TypeEngine
from flyteidl2.core.literals_pb2 import Binary, Literal, Scalar
from omegaconf import MISSING, DictConfig, MissingMandatoryValue, OmegaConf

from flyteplugins.omegaconf.base_transformer import DictConfigTransformer
from flyteplugins.omegaconf.codec import (
    KIND_DICT,
    PAYLOAD_KIND,
    PAYLOAD_MARKER,
    PAYLOAD_SCHEMA,
    PAYLOAD_VALUES,
)

# ── Shared dataclasses ────────────────────────────────────────────────────────


@dataclass
class OptimizerConf:
    lr: float = 0.001
    weight_decay: float = 1e-4


@dataclass
class ModelConf:
    hidden_dim: int = 512
    num_layers: int = 6


@dataclass
class TrainConf:
    optimizer: OptimizerConf = field(default_factory=OptimizerConf)
    model: ModelConf = field(default_factory=ModelConf)
    epochs: int = 10
    batch_size: int = 32


@dataclass
class NetworkConf:
    layer_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu", "sigmoid"])


@dataclass
class RequiredConf:
    data_path: str = MISSING
    epochs: int = 10


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_transformer() -> DictConfigTransformer:
    return DictConfigTransformer()


async def roundtrip(cfg: DictConfig) -> DictConfig:
    t = make_transformer()
    lt = t.get_literal_type(DictConfig)
    literal = await t.to_literal(cfg, DictConfig, lt)
    return await t.to_python_value(literal, DictConfig)


# ── Registration ──────────────────────────────────────────────────────────────


def test_transformer_registered():
    transformer = TypeEngine.get_transformer(DictConfig)
    assert isinstance(transformer, DictConfigTransformer)


def test_get_literal_type_is_struct():
    from flyteidl2.core.types_pb2 import SimpleType

    t = make_transformer()
    lt = t.get_literal_type(DictConfig)
    assert lt.simple == SimpleType.STRUCT


def test_get_literal_type_has_msgpack_annotation():
    from flyte.types._type_engine import CACHE_KEY_METADATA, SERIALIZATION_FORMAT

    t = make_transformer()
    lt = t.get_literal_type(DictConfig)
    annotations = dict(lt.annotation.annotations)
    assert CACHE_KEY_METADATA in annotations
    assert annotations[CACHE_KEY_METADATA][SERIALIZATION_FORMAT] == MESSAGEPACK


# ── Plain dict roundtrips ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plain_dict_flat():
    cfg = OmegaConf.create({"lr": 0.001, "epochs": 10})
    result = await roundtrip(cfg)
    assert isinstance(result, DictConfig)
    assert OmegaConf.get_type(result) is dict
    assert result.lr == 0.001
    assert result.epochs == 10


@pytest.mark.asyncio
async def test_plain_dict_nested():
    cfg = OmegaConf.create(
        {
            "optimizer": {"lr": 0.001, "weight_decay": 1e-4},
            "training": {"epochs": 10, "batch_size": 32},
        }
    )
    result = await roundtrip(cfg)
    assert result.optimizer.lr == 0.001
    assert result.training.batch_size == 32


@pytest.mark.asyncio
async def test_plain_dict_deep_nesting():
    cfg = OmegaConf.create({"a": {"b": {"c": {"d": {"value": 42}}}}})
    result = await roundtrip(cfg)
    assert result.a.b.c.d.value == 42


@pytest.mark.asyncio
async def test_plain_dict_with_list_values():
    cfg = OmegaConf.create(
        {
            "model": {
                "layer_sizes": [64, 128, 256],
                "activations": ["relu", "relu", "sigmoid"],
            },
            "data": {"input_size": [224, 224]},
        }
    )
    result = await roundtrip(cfg)
    assert list(result.model.layer_sizes) == [64, 128, 256]
    assert list(result.model.activations) == ["relu", "relu", "sigmoid"]
    assert list(result.data.input_size) == [224, 224]


@pytest.mark.asyncio
async def test_plain_dict_empty():
    cfg = OmegaConf.create({})
    result = await roundtrip(cfg)
    assert isinstance(result, DictConfig)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_plain_dict_various_value_types():
    cfg = OmegaConf.create(
        {
            "int_val": 42,
            "float_val": 3.14,
            "str_val": "hello",
            "bool_val": True,
            "none_val": None,
        }
    )
    result = await roundtrip(cfg)
    assert result.int_val == 42
    assert result.float_val == pytest.approx(3.14)
    assert result.str_val == "hello"
    assert result.bool_val is True
    assert result.none_val is None


# ── Interpolation ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_interpolation_resolved_at_serialization():
    cfg = OmegaConf.create(
        {
            "base_lr": 0.01,
            "optimizer": {"lr": "${base_lr}"},
        }
    )
    result = await roundtrip(cfg)
    # Interpolation resolved by to_container(resolve=True) — no longer a reference
    assert result.optimizer.lr == 0.01


@pytest.mark.asyncio
async def test_interpolation_cross_key():
    cfg = OmegaConf.create(
        {
            "width": 512,
            "height": "${width}",
            "area": 262144,
        }
    )
    result = await roundtrip(cfg)
    assert result.height == 512


# ── Merged config ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_merged_plain_configs():
    base = OmegaConf.create({"lr": 0.001, "epochs": 10})
    override = OmegaConf.create({"lr": 0.01})
    cfg = OmegaConf.merge(base, override)
    result = await roundtrip(cfg)
    assert result.lr == 0.01
    assert result.epochs == 10


@pytest.mark.asyncio
async def test_merged_structured_with_plain():
    base = OmegaConf.structured(TrainConf())
    override = OmegaConf.create({"optimizer": {"lr": 0.05}, "epochs": 100})
    cfg = OmegaConf.merge(base, override)
    result = await roundtrip(cfg)
    assert OmegaConf.get_type(result) == TrainConf
    assert result.optimizer.lr == 0.05
    assert result.epochs == 100


# ── Structured config roundtrips ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_structured_flat_roundtrip():
    cfg = OmegaConf.structured(TrainConf(epochs=50, batch_size=64))
    result = await roundtrip(cfg)
    assert OmegaConf.get_type(result) == TrainConf
    assert result.epochs == 50
    assert result.batch_size == 64


@pytest.mark.asyncio
async def test_structured_nested_dataclass_roundtrip():
    cfg = OmegaConf.structured(
        TrainConf(
            optimizer=OptimizerConf(lr=0.1, weight_decay=0.01),
            model=ModelConf(hidden_dim=768, num_layers=12),
        )
    )
    result = await roundtrip(cfg)
    assert OmegaConf.get_type(result) == TrainConf
    assert result.optimizer.lr == 0.1
    assert result.optimizer.weight_decay == 0.01
    assert result.model.hidden_dim == 768
    assert result.model.num_layers == 12


@pytest.mark.asyncio
async def test_structured_type_enforced_after_roundtrip():
    """Structured config comes back type-validated — wrong assignment raises."""
    from omegaconf import ValidationError

    cfg = OmegaConf.structured(TrainConf())
    result = await roundtrip(cfg)
    with pytest.raises((ValidationError, Exception)):
        result.optimizer.lr = "not-a-float"


@pytest.mark.asyncio
async def test_structured_with_list_fields():
    cfg = OmegaConf.structured(
        NetworkConf(
            layer_sizes=[256, 128, 64],
            activations=["relu", "relu", "relu"],
        )
    )
    result = await roundtrip(cfg)
    assert OmegaConf.get_type(result) == NetworkConf
    assert list(result.layer_sizes) == [256, 128, 64]
    assert list(result.activations) == ["relu", "relu", "relu"]


@pytest.mark.asyncio
async def test_structured_unknown_class_falls_back_to_plain_dictconfig():
    """If the serialized class can't be imported, falls back to plain DictConfig."""
    t = make_transformer()
    # Craft a payload referencing a non-existent class
    payload = {
        PAYLOAD_MARKER: True,
        PAYLOAD_KIND: KIND_DICT,
        PAYLOAD_SCHEMA: "nonexistent.module.FakeConf",
        PAYLOAD_VALUES: {"lr": 0.001, "epochs": 10},
    }
    binary = Binary(value=msgpack.dumps(payload), tag=MESSAGEPACK)
    result = t.from_binary_idl(binary, DictConfig)
    assert isinstance(result, DictConfig)
    assert OmegaConf.get_type(result) is dict
    assert result.lr == 0.001


# ── MISSING ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_field_serializes_and_raises_on_access():
    """A MISSING field survives roundtrip: serialization succeeds, but accessing the field raises."""
    cfg = OmegaConf.structured(RequiredConf())  # data_path is MISSING

    # Accessing MISSING before roundtrip raises immediately
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.data_path

    # Serialization succeeds (throw_on_missing defaults to False)
    result = await roundtrip(cfg)

    # After roundtrip the structured config is reconstructed, MISSING is still MISSING
    with pytest.raises(MissingMandatoryValue):
        _ = result.data_path


@pytest.mark.asyncio
async def test_missing_field_filled_serializes_fine():
    cfg = OmegaConf.structured(RequiredConf(data_path="/data/imagenet"))
    result = await roundtrip(cfg)
    assert OmegaConf.get_type(result) == RequiredConf
    assert result.data_path == "/data/imagenet"
    assert result.epochs == 10


# ── Wire format ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_literal_is_binary_msgpack():
    t = make_transformer()
    cfg = OmegaConf.create({"lr": 0.001})
    lt = t.get_literal_type(DictConfig)
    literal = await t.to_literal(cfg, DictConfig, lt)
    assert literal.HasField("scalar")
    assert literal.scalar.HasField("binary")
    assert literal.scalar.binary.tag == MESSAGEPACK


@pytest.mark.asyncio
async def test_payload_structure():
    """Serialized payload is a marked DictConfig payload with schema and values."""
    t = make_transformer()
    cfg = OmegaConf.structured(TrainConf())
    lt = t.get_literal_type(DictConfig)
    literal = await t.to_literal(cfg, DictConfig, lt)
    payload = msgpack.loads(literal.scalar.binary.value, strict_map_key=False)
    assert payload[PAYLOAD_MARKER] is True
    assert payload[PAYLOAD_KIND] == KIND_DICT
    assert "TrainConf" in payload[PAYLOAD_SCHEMA]
    assert PAYLOAD_VALUES in payload
    assert payload[PAYLOAD_VALUES]["epochs"] == 10
    assert payload[PAYLOAD_VALUES]["optimizer"][PAYLOAD_MARKER] is True
    assert payload[PAYLOAD_VALUES]["optimizer"][PAYLOAD_KIND] == KIND_DICT
    assert "OptimizerConf" in payload[PAYLOAD_VALUES]["optimizer"][PAYLOAD_SCHEMA]


def test_from_binary_idl_wrong_tag_raises():
    t = make_transformer()
    binary = Binary(value=b"data", tag="json")
    with pytest.raises(TypeError, match="Unsupported binary format"):
        t.from_binary_idl(binary, DictConfig)


@pytest.mark.asyncio
async def test_to_python_value_non_binary_raises():
    t = make_transformer()
    literal = Literal(scalar=Scalar())
    with pytest.raises(TypeError):
        await t.to_python_value(literal, DictConfig)


# ── Task definitions ──────────────────────────────────────────────────────────

_env = flyte.TaskEnvironment(name="omegaconf-dictconfig-test")


@_env.task
async def _identity(cfg: DictConfig) -> DictConfig:
    return cfg


@_env.task
async def _merge(base: DictConfig, override: DictConfig) -> DictConfig:
    return OmegaConf.merge(base, override)


@_env.task
async def _extract_lr(cfg: DictConfig) -> float:
    return float(cfg.optimizer.lr)


@_env.task
async def _config_type_name(cfg: DictConfig) -> str:
    return OmegaConf.get_type(cfg).__name__


# ── Task-based integration tests ──────────────────────────────────────────────


def test_task_plain_dictconfig_roundtrip():
    flyte.init()
    cfg = OmegaConf.create({"optimizer": {"lr": 0.001}, "epochs": 10})
    run = flyte.run(_identity, cfg=cfg)
    result = run.outputs()[0]
    assert isinstance(result, DictConfig)
    assert result.optimizer.lr == pytest.approx(0.001)
    assert result.epochs == 10


def test_task_nested_dictconfig_roundtrip():
    flyte.init()
    cfg = OmegaConf.create(
        {
            "model": {"layers": [256, 128, 64], "dropout": 0.1},
            "training": {"epochs": 50, "lr": 0.01},
        }
    )
    run = flyte.run(_identity, cfg=cfg)
    result = run.outputs()[0]
    assert list(result.model.layers) == [256, 128, 64]
    assert result.training.epochs == 50


def test_task_structured_dictconfig_roundtrip():
    flyte.init()
    cfg = OmegaConf.structured(TrainConf(optimizer=OptimizerConf(lr=0.01), epochs=20))
    run = flyte.run(_identity, cfg=cfg)
    result = run.outputs()[0]
    assert OmegaConf.get_type(result) == TrainConf
    assert result.optimizer.lr == pytest.approx(0.01)
    assert result.epochs == 20


def test_task_structured_config_type_survives():
    flyte.init()
    run = flyte.run(_config_type_name, cfg=OmegaConf.structured(TrainConf()))
    assert run.outputs()[0] == "TrainConf"


def test_task_merge_configs():
    flyte.init()
    base = OmegaConf.create({"optimizer": {"lr": 0.001}, "epochs": 10})
    override = OmegaConf.create({"optimizer": {"lr": 0.01}, "epochs": 20})
    run = flyte.run(_merge, base=base, override=override)
    result = run.outputs()[0]
    assert result.optimizer.lr == pytest.approx(0.01)
    assert result.epochs == 20


def test_task_extract_value_from_nested_config():
    flyte.init()
    cfg = OmegaConf.structured(TrainConf(optimizer=OptimizerConf(lr=0.005)))
    run = flyte.run(_extract_lr, cfg=cfg)
    assert run.outputs()[0] == pytest.approx(0.005)


def test_task_missing_field_survives_roundtrip():
    """MISSING config passes through a task; the field still raises on access after return."""
    flyte.init()
    cfg = OmegaConf.structured(RequiredConf())  # data_path is MISSING
    run = flyte.run(_identity, cfg=cfg)
    result = run.outputs()[0]
    with pytest.raises(MissingMandatoryValue):
        _ = result.data_path
    assert result.epochs == 10


def test_task_filled_required_config():
    flyte.init()
    cfg = OmegaConf.structured(RequiredConf(data_path="/data/imagenet"))
    run = flyte.run(_identity, cfg=cfg)
    result = run.outputs()[0]
    assert result.data_path == "/data/imagenet"
