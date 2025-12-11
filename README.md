# Flyte 2 SDK 🚀

**Type-safe, distributed orchestration of agents, ML pipelines, and more — in pure Python with async/await or sync!**

[![Version](https://img.shields.io/pypi/v/flyte?label=version&color=blue)](https://pypi.org/project/flyte/)
[![Python](https://img.shields.io/pypi/pyversions/flyte?color=brightgreen)](https://pypi.org/project/flyte/)
[![License](https://img.shields.io/badge/license-Apache%202.0-orange)](LICENSE)

> ⚡ **Pure Python workflows** • 🔄 **Async-first parallelism** • 🛠️ **Zero DSL constraints** • 📊 **Sub-task observability**

## 🌍 Ecosystem & Resources

- **📖 Documentation**: [Docs Link](https://www.union.ai/docs/v2/flyte/user-guide/)
- **▶️ Getting Started**: [Docs Link](https://www.union.ai/docs/v2/flyte/user-guide/getting-started/)
- **💬 Community**: [Slack](https://slack.flyte.org/) | [GitHub Discussions](https://github.com/flyteorg/flyte/discussions)  
- **🎓 Examples**: [GitHub Examples](https://github.com/flyteorg/flyte-sdk/tree/main/examples)
- **🐛 Issues**: [Bug Reports](https://github.com/flyteorg/flyte/issues)

## What is Flyte 2?

Flyte 2 represents a fundamental shift from constrained domain-specific languages to **pure Python workflows**. Write data pipelines, ML training jobs, and distributed compute exactly like you write Python—because it *is* Python.

```python
import flyte

env = flyte.TaskEnvironment("hello_world")

@env.task
async def process_data(data: list[str]) -> list[str]:
    # Use any Python construct: loops, conditionals, try/except
    results = []
    for item in data:
        if len(item) > 5:
            results.append(await transform_item(item))
    return results

@env.task
async def transform_item(item: str) -> str:
    return f"processed: {item.upper()}"

if __name__ == "__main__":
    flyte.init()
    result = flyte.run(process_data, data=["hello", "world", "flyte"])
```

## 🌟 Why Flyte 2?

| Feature Highlight | Flyte 1 | Flyte 2 |
|-| ------- | ------- |
| **No More Workflow DSL** | ❌ `@workflow` decorators with Python subset limitations | ✅ **Pure Python**: loops, conditionals, error handling, dynamic structures |
| **Async-First Parallelism** | ❌ Custom `map()` functions and workflow-specific parallel constructs | ✅ **Native `asyncio`**: `await asyncio.gather()` for distributed parallel execution |
| **Fine-Grained Observability** | ❌ Task-level logging only | ✅ **Function-level tracing** with `@flyte.trace` for sub-task checkpoints |

## 🚀 Quick Start

### Installation

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv && source .venv/bin/activate

# Install Flyte 2 (beta)
uv pip install --prerelease=allow flyte
```

### Your First Workflow

```python
# hello.py
# /// script
# requires-python = ">=3.10"
# dependencies = ["flyte>=2.0.0b0"]
# ///

import flyte

env = flyte.TaskEnvironment(
    name="hello_world",
    resources=flyte.Resources(memory="250Mi")
)

@env.task
def calculate(x: int) -> int:
    return x * 2 + 5

@env.task
async def main(numbers: list[int]) -> float:
    # Parallel execution across distributed containers
    results = await asyncio.gather(*[
        calculate.aio(num) for num in numbers
    ])
    return sum(results) / len(results)

if __name__ == "__main__":
    flyte.init_from_config("config.yaml")
    run = flyte.run(main, numbers=list(range(10)))
    print(f"Result: {run.result}")
    print(f"View at: {run.url}")
```

```bash
# Run locally, execute remotely
uv run --prerelease=allow hello.py
```

## 🏗️ Core Concepts

### **TaskEnvironments**: Container Configuration Made Simple

```python
# Group tasks with shared configuration
env = flyte.TaskEnvironment(
    name="ml_pipeline",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "torch", "pandas", "scikit-learn"
    ),
    resources=flyte.Resources(cpu=4, memory="8Gi", gpu=1),
    reusable=flyte.ReusePolicy(replicas=3, idle_ttl=300)
)

@env.task
def train_model(data: flyte.File) -> flyte.File:
    # Runs in configured container with GPU access
    pass

@env.task
def evaluate_model(model: flyte.File, test_data: flyte.File) -> dict:
    # Same container configuration, different instance
    pass
```

### **Pure Python Workflows**: No More DSL Constraints

```python
@env.task
async def dynamic_pipeline(config: dict) -> list[str]:
    results = []

    # ✅ Use any Python construct
    for dataset in config["datasets"]:
        try:
            # ✅ Native error handling
            if dataset["type"] == "batch":
                result = await process_batch(dataset)
            else:
                result = await process_stream(dataset)
            results.append(result)
        except ValidationError as e:
            # ✅ Custom error recovery
            result = await handle_error(dataset, e)
            results.append(result)

    return results
```

### **Async Parallelism**: Distributed by Default

```python
@env.task
async def parallel_training(hyperparams: list[dict]) -> dict:
    # Each model trains on separate infrastructure
    models = await asyncio.gather(*[
        train_model.aio(params) for params in hyperparams
    ])

    # Evaluate all models in parallel
    evaluations = await asyncio.gather(*[
        evaluate_model.aio(model) for model in models
    ])

    # Find best model
    best_idx = max(range(len(evaluations)),
                   key=lambda i: evaluations[i]["accuracy"])
    return {"best_model": models[best_idx], "accuracy": evaluations[best_idx]}
```

## 🎯 Advanced Features

### **Sub-Task Observability with Tracing**

```python
@flyte.trace
async def expensive_computation(data: str) -> str:
    # Function-level checkpointing - recoverable on failure
    result = await call_external_api(data)
    return process_result(result)

@env.task(cache=flyte.Cache(behavior="auto"))
async def main_task(inputs: list[str]) -> list[str]:
    results = []
    for inp in inputs:
        # If task fails here, it resumes from the last successful trace
        result = await expensive_computation(inp)
        results.append(result)
    return results
```

### **Remote Task Execution**

```python
import flyte.remote

# Reference tasks deployed elsewhere
torch_task = flyte.remote.Task.get("torch_env.train_model", auto_version="latest")
spark_task = flyte.remote.Task.get("spark_env.process_data", auto_version="latest")

@env.task
async def orchestrator(raw_data: flyte.File) -> flyte.File:
    # Execute Spark job on big data cluster
    processed = await spark_task(raw_data)

    # Execute PyTorch training on GPU cluster
    model = await torch_task(processed)

    return model
```

## 📊 Native Jupyter Integration

Run and monitor workflows directly from notebooks:

```python
# In Jupyter cell
import flyte

flyte.init_from_config()
run = flyte.run(my_workflow, data=large_dataset)

# Stream logs in real-time
run.logs.stream()

# Get outputs when complete
results = run.wait()
```

## 🔧 Configuration & Deployment

### Configuration File

```yaml
# config.yaml
endpoint: https://my-flyte-instance.com
project: ml-team
domain: production
image:
  builder: local
  registry: ghcr.io/my-org
auth:
  type: oauth2
```

### Deploy and Run

```bash
# Deploy tasks to remote cluster
flyte deploy my_workflow.py

# Run deployed workflow
flyte run my_workflow --input-file params.json

# Monitor execution
flyte logs <execution-id>
```

## Migration from Flyte 1

| Flyte 1 | Flyte 2 |
|---------|---------|
| `@workflow` + `@task` | `@env.task` only |
| `flytekit.map()` | `await asyncio.gather()` |
| `@dynamic` workflows | Regular `@env.task` with loops |
| `flytekit.conditional()` | Python `if/else` |
| `LaunchPlan` schedules | `@env.task(on_schedule=...)` |
| Workflow failure handlers | Python `try/except` |

## 🤝 Contributing

We welcome contributions! Whether it's:

- 🐛 **Bug fixes**
- ✨ **New features**
- 📚 **Documentation improvements**
- 🧪 **Testing enhancements**

### Setup & Iteration Cycle
To get started, make sure you start from a new virtual environment and install this package in editable mode with any of the supported Python versions, from 3.10 to 3.13.

```bash
uv venv --python 3.13
uv pip install -e .
```

Besides from picking up local code changes, installing the package in editable mode
also changes the definition of the default `Image()` object to use a locally
build wheel. You will need to build said wheel by yourself though, with the `make dist` target.

```bash
make dist
python maint_tools/build_default_image.py
```
You'll need to have a local docker daemon running for this. The build script does nothing
more than invoke the local image builder, which will create a buildx builder named `flytex` if not present. Note that only members of the `Flyte Maintainers` group has
access to push to the default registry. If you don't have access, please make sure to
specify the registry and name to the build script.

```bash
python maint_tools/build_default_image.py --registry ghcr.io/my-org --name my-flyte-image
```

## 📄 License

Flyte 2 is licensed under the [Apache 2.0 License](LICENSE).

## Developing the Core Controller

Create a separate virtual environment for the Rust contoller inside the rs_controller folder. The reason for this is
because the rust controller should be a separate pypi package. The reason it should be a separate pypi package is that
including it into the main SDK as a core component means the entire build toolchain for the SDK will need to become
rust/maturin based. We should probably move to this model in the future though.

Keep important dependencies the same though, namely flyteidl2.

The following instructions are for helping to build the default multi-arch image. Each architecture needs a different wheel. Each wheel needs to be built by a different docker image.

### Setup Builders
`cd` into `rs_controller` and run `make build-builders`. This will build the builder images once, so you can keep using them as the rust code changes.

### Iteration Cycle
Make sure you have `CLOUD_REPO=/Users/<username>/go/src/github.com/unionai/cloud` exported and checked out to a branch that has the latest prost generated code. Delete this comment and update make target in the future if it gets merged/published.

Then run `make build-wheels`.

`cd` back up to the root folder of this project and proceed with
```bash
make dist
python maint_tools/build_default_image.py
```

To install the wheel locally for testing, use the following command with your venv active.
```bash
uv pip install --find-links ./rs_controller/dist --no-index --force-reinstall flyte_controller_base
```
Repeat this process to iterate - build new wheels, force reinstall the controller package.

### Build Configuration Summary

In order to support both Rust crate publication and Python wheel distribution, we have
to sometimes use and sometimes not use the 'pyo3/extension-module' feature. To do this, this
project's Cargo.toml itself can toggle this on and off.

  [features]
  default = ["pyo3/auto-initialize"]     # For Rust crate users (links to libpython)
  extension-module = ["pyo3/extension-module"]  # For Python wheels (no libpython linking)

The cargo file contains

  # Cargo.toml
  [lib]
  crate-type = ["rlib", "cdylib"]  # Support both Rust and Python usage

When using 'default', 'auto-initialize' is turned on, which requires linking to libpython, which exists on local Mac so
this works nicely. It is not available in manylinux however, so trying to build with this feature in a manylinux docker
image will fail. But that's okay, because the purpose of the manylinux container is to build wheels,
and for wheels, we need the 'extension-module' feature, which disables linking to libpython.

The key insight: auto-initialize is for embedding Python in Rust (needs libpython), while
extension-module is for extending Python with Rust (must NOT link libpython for portability).

This setup makes it possible to build wheels and also run Rust binaries with `cargo run --bin`. 

(not sure if this is needed)
  # pyproject.toml
  [tool.maturin]
  features = ["extension-module"]  # Tells maturin to use extension-module feature
