# Sleep Plugin

`flyteplugins-sleep` lets you author Flyte tasks with `TaskTemplate.type = "core-sleep"`.

This is useful when a backend enables the in-process core sleep plugin and you want to keep actions running inside
leaseworker without creating task pods.

## Installation

```bash
pip install flyteplugins-sleep
```

## Usage

```python
from datetime import timedelta
import pathlib

import flyte
from flyteplugins.sleep import Sleep

# The core-sleep plugin executes in leaseworker, so the remote task image does
# not need flyteplugins-sleep installed.
image = flyte.Image.from_debian_base(python_version=(3, 12))

sleep_env = flyte.TaskEnvironment(
    name="sleep-env",
    image=image,
    plugin_config=Sleep(),
)


@sleep_env.task
async def sleep_for(duration: timedelta) -> None:
    return None


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(sleep_for, duration=timedelta(seconds=60))
    print(run.name)
    print(run.url)
```

## Behavior

- The backend must have the `core-sleep` plugin enabled.
- `flyteplugins-sleep` only needs to be installed in the local submission environment.
- The remote task image does not need `flyteplugins-sleep`, because `core-sleep` does not execute the Python container.
- Remote execution is backend-defined. The Python function body is not executed remotely.
- `core-sleep` requires exactly one `timedelta` input.
- `core-sleep` does not support outputs.
