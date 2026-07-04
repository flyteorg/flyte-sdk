# Flyte Trackio Plugin

Native Flyte support for Trackio experiment tracking.

This plugin makes it easy to manage Trackio runs inside Flyte tasks, log metrics from within task code or callbacks, and expose a Trackio dashboard link in the Flyte UI.

## What it provides

- `@trackio_init` decorator to initialize and manage a Trackio run for a Flyte task.
- `trackio_config(...)` helper to store Trackio initialization settings in Flyte `custom_context`.
- `get_trackio_run()` to access the active Trackio run from task code or callback code.
- Optional Flyte task link support via the `Trackio` link class.

## Installation

```bash
pip install flyteplugins-trackio
```

## Quick start

```python
import flyte
from flyteplugins.trackio import (
    get_trackio_run,
    trackio_config,
    trackio_init,
)

env = flyte.TaskEnvironment(name="trackio")

@trackio_init
@env.task
def train() -> dict[str, float]:
    run = get_trackio_run()
    run.log({"accuracy": 0.92})
    return {"accuracy": 0.92}

cfg = trackio_config(
    project="my-project",
    space_id="my-org/my-trackio-space",
    bucket_id="my-bucket",
    auto_log_cpu=True,
)

flyte.with_runcontext(custom_context=cfg.to_dict()).run(train)
```

## Key APIs

### `trackio_init`

Decorates a Flyte task or plain Python function to create a Trackio run for the decorated execution.

- Works with Flyte task functions.
- Works with plain synchronous or asynchronous functions outside of Flyte.
- Reuses an existing Trackio run if one is already present in the Flyte context.

### `trackio_config(...)`

Returns a configuration object with Trackio initialization options such as:

- `project`
- `name`
- `group`
- `space_id`
- `dataset_id`
- `bucket_id`
- `server_url`
- `config`
- `resume`
- `auto_log_gpu`
- `gpu_log_interval`
- `auto_log_cpu`
- `cpu_log_interval`

Use `cfg.to_dict()` to store these settings in Flyte `custom_context`, then run the task with `flyte.with_runcontext(...)`.

### `get_trackio_run()`

Returns the active Trackio run from the current Flyte context, or falls back to Trackio's global active run when running outside of Flyte.

### `Trackio`

A Flyte `Link` implementation that resolves a Trackio dashboard URL based on:

- explicit `server_url`
- `space_id` for Hugging Face Space deployments
- plugin `project`/context values

This link can be attached automatically by `trackio_init` for Flyte tasks.

## Usage notes

- The plugin manages the Trackio run lifecycle automatically: `trackio_init` creates and finishes the run around the decorated execution.
- Custom context values from `trackio_config` are merged with decorator-level overrides.
- Use `get_trackio_run()` in callbacks or training loops to log metrics incrementally.

## Examples

See `[distilbert_text_classification]()` and `[vit_image_classification.py]()` for full usage patterns with Hugging Face training and callback-based metric logging.