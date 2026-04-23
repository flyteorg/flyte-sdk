# Flyte 2 Features

Flyte 2 is a fundamental shift from constrained DSLs to **pure Python**. Write pipelines, serve models, and orchestrate agents exactly like you write Python — because it *is* Python.

## Highlights

### Pure Python Workflows

No more workflow DSL. Use loops, conditionals, try/except, and any Python construct:

```python
@env.task
async def dynamic_pipeline(config: dict) -> list[str]:
    results = []
    for dataset in config["datasets"]:
        try:
            if dataset["type"] == "batch":
                result = await process_batch(dataset)
            else:
                result = await process_stream(dataset)
            results.append(result)
        except ValidationError as e:
            results.append(await handle_error(dataset, e))
    return results
```

### App Serving

Serve models and apps as first-class Flyte deployments:

```python
from fastapi import FastAPI
from flyte.app.extras import FastAPIAppEnvironment
import flyte

app = FastAPI()
env = FastAPIAppEnvironment(
    name="hello-api",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi", "uvicorn"
    ),
)

@app.get("/predict")
async def predict(x: float) -> dict:
    return {"result": x * 2 + 5}
```

```bash
flyte serve serving.py env
```

### Async Parallelism

Native `asyncio` for distributed parallel execution — no custom map functions:

```python
@env.task
async def parallel_training(hyperparams: list[dict]) -> dict:
    models = await asyncio.gather(*[
        train_model.aio(params) for params in hyperparams
    ])
    evaluations = await asyncio.gather(*[
        evaluate_model.aio(model) for model in models
    ])
    best_idx = max(range(len(evaluations)),
                   key=lambda i: evaluations[i]["accuracy"])
    return {"best_model": models[best_idx], "accuracy": evaluations[best_idx]}
```

## Feature Reference

| Feature | What it does | Why you need it | Example |
|---|---|---|---|
| **Task Environments** | Group tasks with shared container config, resources, and images | Define infrastructure once, reuse across tasks | [image/container_images.py](examples/image/container_images.py) |
| **Reusable Containers** | Keep containers warm between task invocations | Eliminate cold-start latency for iterative workloads | [reuse/reusable.py](examples/reuse/reusable.py) |
| **Caching** | Content-based or version-based task result caching | Skip redundant computation, save time and cost | [caching/content_based_caching.py](examples/caching/content_based_caching.py) |
| **Tracing** | Function-level checkpointing with `@flyte.trace` | Resume from the last successful step on failure | [basics/hello.py](examples/basics/hello.py) |
| **File & Dir I/O** | `flyte.io.File` and `flyte.io.Dir` for large data transfer | Move large artifacts between tasks without manual S3/GCS plumbing | [basics/file.py](examples/basics/file.py) |
| **Streaming** | Stream results as they become available | Process outputs incrementally instead of waiting for completion | [streaming/basic_as_completed.py](examples/streaming/basic_as_completed.py) |
| **GPU / Accelerators** | Request GPUs, TPUs, Trainium, Habana | Run training and inference on specialized hardware | [accelerators/gpu.py](examples/accelerators/gpu.py) |
| **Triggers** | Schedule tasks on time or events | Automate recurring pipelines and event-driven workflows | [triggers/basic.py](examples/triggers/basic.py) |
| **Connectors** | BigQuery, Snowflake, Databricks integrations | Query external data systems directly from tasks | [connectors/snowflake_example.py](examples/connectors/snowflake_example.py) |
| **Reports** | Interactive data visualizations and dashboards | Generate rich HTML reports from task outputs | [reports/dataframe_report.py](examples/reports/dataframe_report.py) |
| **GenAI Agents** | Build and orchestrate AI agents | Run LLM-powered agents with tool use and handoffs | [genai/hello_agent.py](examples/genai/hello_agent.py) |
| **Apps** | Serve FastAPI, Streamlit, Gradio, Panel apps | Deploy and scale web apps alongside your pipelines | [apps/single_script_fastapi.py](examples/apps/single_script_fastapi.py) |
| **Remote Tasks** | Call tasks deployed in other environments | Compose pipelines across teams and infrastructure | [remote_management/remote_validate.py](examples/remote_management/remote_validate.py) |
| **Plugins** | Spark, Ray, Dask, PyTorch distributed | Run workloads on specialized distributed compute frameworks | [plugins/spark_example.py](examples/plugins/spark_example.py) |
| **Higher-Order Patterns** | Circuit breakers, auto-batching, OOM retriers | Production resilience patterns out of the box | [higher_order_patterns/circuit_breaker.py](examples/higher_order_patterns/circuit_breaker.py) |
| **Volumes** | Mount GCSFuse and other volumes into tasks | Access cloud storage as a local filesystem | [volumes/gcsfuse_example.py](examples/volumes/gcsfuse_example.py) |

## CLI

The Flyte CLI follows a **verb noun** structure. Full reference: [CLI Docs](https://www.union.ai/docs/v2/union/api-reference/flyte-cli/)

```bash
flyte run hello.py main --numbers '[1,2,3]'     # Run a task
flyte serve serving.py env                        # Serve an app
flyte deploy my_workflow.py                       # Deploy environments
flyte build my_workflow.py --push                 # Build and push images
flyte get logs <run-name>                         # Get logs for a run
flyte abort run <run-name>                        # Abort a run
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
