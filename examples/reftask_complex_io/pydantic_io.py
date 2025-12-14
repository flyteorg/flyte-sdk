from typing import Any

from pydantic import BaseModel

import flyte

env = flyte.TaskEnvironment(name="py-io")


class MyNestedModel(BaseModel):
    x: int


class MyInput(BaseModel):
    x: int
    y: int
    m: MyNestedModel


@env.task
async def dynamic_wf(
    workflow_config: MyInput,  # Pydantic BaseModel
    user_params: dict[str, Any],
) -> None:
    print(workflow_config, flush=True)
    print(user_params, flush=True)


env2 = flyte.TaskEnvironment(name="py-io-2")


@env2.task
async def main():
    import flyte.remote

    ref_task = flyte.remote.Task.get("py-io.dynamic_wf", auto_version="latest")
    await ref_task(workflow_config={"x": 1, "y": 2, "m": {"x": 1}}, user_params={"y": dynamic_wf})


def run_direct():
    r = flyte.run(dynamic_wf, MyInput(x=1, y=2, m=MyNestedModel(x=1)), {"x": dynamic_wf})
    print(r.url)


def run_ref():
    import flyte.remote

    flyte.deploy(env)
    ref_task = flyte.remote.Task.get("py-io.dynamic_wf", auto_version="latest")
    r = flyte.run(ref_task, workflow_config={"x": 1, "y": 2, "m": {"x": 1}}, user_params={"y": dynamic_wf})
    print(r.url)


def run_wrapper():
    import flyte.remote

    flyte.deploy(env)
    r = flyte.run(main)
    print(r.url)


if __name__ == "__main__":
    flyte.init_from_config()
    run_direct()
    run_ref()
    run_wrapper()
