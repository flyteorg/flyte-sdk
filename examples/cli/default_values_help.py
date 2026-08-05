"""
Repro for partial ``--inputs`` on deployed tasks with Pydantic field defaults.

Mirrors the customer report: a single ``inputs`` argument typed as a Pydantic
model whose fields have defaults, with the whole model as the task-arg default.

Deploy and run (adjust project/domain/config to your cluster):

    flyte deploy examples/cli/default_values_help.py env

    flyte run deployed-task deploy_task_test.hello_flyte_task --help
    # expect: --inputs ... [default: {"font": "standard", "message": "hello, flyte"}]

    # partial override — font should default to "standard"
    flyte run deployed-task deploy_task_test.hello_flyte_task --inputs '{"message":"hello, niels"}'

    # no --inputs — full task default
    flyte run deployed-task deploy_task_test.hello_flyte_task

Local (no deploy):

    flyte run examples/cli/deploy_task_test.py hello_flyte_task --inputs '{"message":"hello, niels"}'
"""

from __future__ import annotations

from pydantic import BaseModel, Field

import flyte

env = flyte.TaskEnvironment(name="deploy_task_test", image=flyte.Image.from_debian_base(python_version=(3, 12)))


class HelloFlyteInputManifest(BaseModel):
    message: str = Field(default="hello, flyte")
    font: str = Field(default="standard")


class HelloFlyteManifest(BaseModel):
    message: str
    font: str


@env.task
def hello_flyte_task(
    inputs: HelloFlyteInputManifest = HelloFlyteInputManifest(),
) -> HelloFlyteManifest:
    print(f"message={inputs.message!r} font={inputs.font!r}")
    return HelloFlyteManifest(message=inputs.message, font=inputs.font)


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(hello_flyte_task, inputs=HelloFlyteInputManifest(message="hello, niels"))
    print(run.url)
    run.wait()
    print(run.outputs())
