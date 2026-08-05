# Echo Plugin

`flyteplugins-echo` lets you author Flyte tasks with `TaskTemplate.type = "echo"`.

This is useful when a backend already enables the in-process `echo` plugin and you want to stress the
leaseworker/leasor execution path without creating task pods.

## Installation

```bash
pip install flyteplugins-echo
```

## Usage

```python
import pathlib

import flyte
from flyteplugins.echo import Echo

# The echo plugin executes in leaseworker, so the remote task image does not
# need flyteplugins-echo installed.
image = flyte.Image.from_debian_base(python_version=(3, 12))

echo_env = flyte.TaskEnvironment(
    name="echo-env",
    image=image,
    plugin_config=Echo(),
)


@echo_env.task
async def echo_identity(message: str) -> str:
    return message


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(echo_identity, message="hello")
    print(run.name)
    print(run.url)
```

## Behavior

- The backend must have the `echo` plugin enabled.
- `flyteplugins-echo` only needs to be installed in the local submission environment.
- The remote task image does not need `flyteplugins-echo`, because `echo` does not execute the Python container.
- Remote execution is backend-defined. The Python function body is not executed remotely.
- `() -> None` tasks succeed without outputs.
- Single-input/single-output tasks behave like an identity task because the backend copies the input literal
  to the output literal.
- Sleep duration, if configured by the backend, is global backend config today and not per-task SDK config.
