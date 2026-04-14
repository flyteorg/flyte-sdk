# flyteplugins-omegaconf

Enables [OmegaConf](https://omegaconf.readthedocs.io/) `DictConfig` and `ListConfig` as typed inputs and outputs for Flyte tasks.

## Installation

```bash
pip install flyteplugins-omegaconf
```

Installing the package automatically registers the `DictConfig` and `ListConfig` transformers with Flyte's TypeEngine via the `flyte.plugins.types` entry point.

## Usage

### DictConfig as task inputs and outputs

```python
import flyte
from omegaconf import DictConfig, OmegaConf

env = flyte.TaskEnvironment(name="training", image=...)

@env.task
async def preprocess(cfg: DictConfig) -> DictConfig:
    return OmegaConf.merge(cfg, {"data": {"normalized": True}})

@env.task
async def train(cfg: DictConfig) -> float:
    return run_experiment(cfg.optimizer.lr, cfg.training.epochs)

@env.task
async def pipeline() -> float:
    cfg = OmegaConf.create({"optimizer": {"lr": 0.001}, "training": {"epochs": 10}})
    processed = await preprocess(cfg)
    return await train(processed)
```

### ListConfig as task inputs and outputs

```python
from omegaconf import ListConfig, OmegaConf

@env.task
async def build_lr_schedule(base_lr: float, num_stages: int) -> ListConfig:
    return OmegaConf.create([base_lr * (0.5 ** i) for i in range(num_stages)])

@env.task
async def train_with_schedule(cfg: DictConfig, lr_schedule: ListConfig) -> float:
    final_lr = float(lr_schedule[-1])
    ...
```

## Ways to construct a DictConfig

All of the following are valid ways to create a `DictConfig` to pass to a task:

### 1. From a plain dict

```python
cfg = OmegaConf.create({"optimizer": {"lr": 0.001}, "training": {"epochs": 10}})
flyte.run(train, cfg=cfg)
```

### 2. From a YAML file

```python
cfg = OmegaConf.load("config.yaml")
flyte.run(train, cfg=cfg)
```

### 3. From a typed dataclass (structured config)

```python
from dataclasses import dataclass, field
from omegaconf import OmegaConf

@dataclass
class OptimizerConf:
    lr: float = 0.001
    weight_decay: float = 1e-4

@dataclass
class TrainConf:
    optimizer: OptimizerConf = field(default_factory=OptimizerConf)
    epochs: int = 10

cfg = OmegaConf.structured(TrainConf())
flyte.run(train, cfg=cfg)
```

Structured configs provide **type validation at assignment time**: `cfg.optimizer.lr = "oops"` raises `omegaconf.ValidationError`.

### 4. Merging base config with overrides

```python
base = OmegaConf.load("config.yaml")
override = OmegaConf.create({"optimizer": {"lr": 0.01}})
cfg = OmegaConf.merge(base, override)
flyte.run(train, cfg=cfg)
```

### 5. Structured config with MISSING required fields

```python
from omegaconf import MISSING

@dataclass
class TrainConf:
    data_path: str = MISSING  # must be set before accessing
    epochs: int = 10

# Pass with MISSING still unset — serialization succeeds
cfg = OmegaConf.structured(TrainConf())
flyte.run(train, cfg=cfg)

# Or fill it before passing
cfg = OmegaConf.structured(TrainConf(data_path="/data/imagenet"))
flyte.run(train, cfg=cfg)
```

A config with an unset `MISSING` field serializes and deserializes successfully — the `MISSING` sentinel is preserved through the wire format. Accessing the field raises `MissingMandatoryValue`, so the task will fail if it tries to read an unfilled field.

## Structured config deserialization

When a `DictConfig` is deserialized in a receiving task, the plugin uses **Auto mode**: it attempts to reconstruct the original dataclass-backed config, and falls back to a plain `DictConfig` if the class is not importable in the receiving task's environment.

```python
# Task A produces a structured config
cfg = OmegaConf.structured(TrainConf(lr=0.01))
# serialized payload: {"base_dataclass": "mymodule.TrainConf", "values": {...}}

# Task B receives it
async def task_b(cfg: DictConfig) -> ...:
    # If TrainConf is importable: cfg is a TrainConf-backed DictConfig (type-validated)
    # If TrainConf is not importable: cfg is a plain DictConfig (no schema)
    OmegaConf.get_type(cfg)  # TrainConf or dict
```

To ensure structured configs survive task hops, make sure the dataclass is defined in a module importable by all tasks in the pipeline.

## Wire format

Both `DictConfig` and `ListConfig` are serialized as MessagePack binaries with tag `"msgpack"`:

```
Literal(scalar=Scalar(binary=Binary(value=<msgpack bytes>, tag="msgpack")))
```

**DictConfig payload** (msgpack-encoded dict):

```json
{
  "base_dataclass": "mymodule.TrainConf",
  "values": { "optimizer": { "lr": 0.001 }, "training": { "epochs": 10 } }
}
```

For plain dict-backed configs, `base_dataclass` is `"builtins.dict"`.

**ListConfig payload** (msgpack-encoded list):

```json
[0.001, 0.01, 0.1]
```

OmegaConf variable interpolations are **resolved** at serialization time (`resolve=True`). The wire representation always contains concrete values.

## Limitations

- **Structured config schema strictness**: merging keys that don't exist as dataclass fields raises an error. Only declare structured configs when all possible keys are known upfront.
- **`MISSING` fields**: a `DictConfig` with unset `MISSING` fields serializes fine — the sentinel is preserved on the wire and accessing it still raises `MissingMandatoryValue`. However, in plain dict mode (when the originating dataclass is not importable in the receiving task), the field's type annotation is lost: the node becomes an `AnyNode` instead of the declared type (e.g. `StringNode`). In Auto mode, the schema is recovered from the dataclass, so the annotation is preserved.
- **ListConfig structured configs**: `ListConfig` always round-trips as a plain `ListConfig` — there is no structured (typed-element) ListConfig support.
- **Key types**: OmegaConf enforces string keys for `DictConfig`; integer-keyed dicts are not supported.
- **Class importability**: structured config reconstruction requires the dataclass to be importable in the receiving task. If it is not, the config falls back to a plain `DictConfig` (Auto mode).

## Examples

See the [`examples/`](examples/) directory:

- [`example_dictconfig.py`](examples/example_dictconfig.py) — plain dict configs, nested, interpolation, merging
- [`example_structured_config.py`](examples/example_structured_config.py) — structured configs, type validation, MISSING fields, config resolution
- [`example_listconfig.py`](examples/example_listconfig.py) — numeric lists, nested lists, list of dicts, LR schedules
- [`example_pipeline.py`](examples/example_pipeline.py) — multi-task pipeline with DictConfig and ListConfig flowing between tasks
