"""
Example: tasks with default input values, exercised through the CLI.

Covers the three default-shaping patterns that the SDK supports for a task
input:

    (1) Pydantic model with field defaults, passed as a default task input
    (2) Frozen dataclass with field defaults, passed as a default task input
    (3) Primitive task args with function-level defaults

To exercise the deployed-task path end-to-end, deploy this file and then
invoke each task via ``flyte run deployed-task`` with no arguments — relying
entirely on the defaults:

    flyte deploy examples/cli/default_values.py env
    flyte run deployed-task default_values_cli.task_pydantic_default
    flyte run deployed-task default_values_cli.task_dataclass_default
    flyte run deployed-task default_values_cli.task_primitive_defaults

``--help`` on any of these should render the actual default value (e.g.
``[default: standard]``), not the ``_has_default`` sentinel class.

You can also run locally (no backend required) with:

    flyte run --local examples/cli/default_values_cli.py task_primitive_defaults
    flyte run --local examples/cli/default_values_cli.py task_pydantic_default
    flyte run --local examples/cli/default_values_cli.py task_dataclass_default
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import pydantic
from pydantic import Field

import flyte

env = flyte.TaskEnvironment(name="default_values_cli")


# ---------------------------------------------------------------------------
# Case (1): Pydantic model with field defaults, given as the task arg default.
# ---------------------------------------------------------------------------


class PydanticInput(pydantic.BaseModel):
    target_revision: Annotated[str, Field(default="head", description="git ref to inspect")] = "head"
    repo: str = "flyteorg/flyte"


@env.task
def task_pydantic_default(inputs: PydanticInput = PydanticInput()) -> str:
    msg = f"[pydantic] repo={inputs.repo} target_revision={inputs.target_revision}"
    print(msg)
    return msg


# ---------------------------------------------------------------------------
# Case (2): Frozen dataclass with field defaults, given as the task arg.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataclassInput:
    message: str = "hello"
    font: str = "standard"


@env.task
def task_dataclass_default(inputs: DataclassInput = DataclassInput()) -> str:
    msg = f"[dataclass] message={inputs.message} font={inputs.font}"
    print(msg)
    return msg


# ---------------------------------------------------------------------------
# Case (3): Primitive args with function-level defaults — the exact shape from
# the customer's `hello_flyte_task` repro.
# ---------------------------------------------------------------------------


@env.task
def task_primitive_defaults(message: str = "hello, flyte", font: str = "standard") -> str:
    msg = f"[primitive] message={message} font={font}"
    print(msg)
    return msg


if __name__ == "__main__":
    flyte.init_from_config()

    print("Running task_pydantic_default")
    run = flyte.run(task_pydantic_default)
    print(run.url)
    run.wait()
    print(run.outputs())

    print("Running task_dataclass_default")
    run = flyte.run(task_dataclass_default)
    print(run.url)
    run.wait()
    print(run.outputs())

    print("Running task_primitive_defaults")
    run = flyte.run(task_primitive_defaults)
    print(run.url)
    run.wait()
    print(run.outputs())
