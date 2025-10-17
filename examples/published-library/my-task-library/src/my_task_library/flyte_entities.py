from typing import List

import flyte

library_environment = flyte.TaskEnvironment(
    name="my-task-library-env",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)


@library_environment.task
async def library_child_task(data: str, lt: List[int]) -> str:
    print("Running library_child_task!")
    return f"Hello {data} {lt}"


@library_environment.task
async def library_parent_task(data: str = "default string", n: int = 3) -> str:
    print(f"Hello from library_parent_task! - {flyte.ctx().action}")
    return await library_child_task(data=data, lt=list(range(n)))


