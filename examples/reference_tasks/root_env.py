"""
This example shows how to use the root environment to run tasks that are defined in other environments
that have been deployed. They are imported using the `flyte.remote.Task.get` method, which fetches the task definitions
from the Flyte control plane, at runtime / deploy time.

This example also demonstrates how "DataClasses" or "pydantic models" can be used as inputs and outputs of tasks,
whose types are faked in the Flyte SDK to allow for easy access to attributes, even though they are not available
in the runtime environment.

"""

import flyte.remote

env = flyte.TaskEnvironment(name="root")

torch_task = flyte.remote.Task.get("torch_env.torch_task", auto_version="latest")

spark_task = flyte.remote.Task.get("spark_env.spark_task", auto_version="latest")
spark_task2 = flyte.remote.Task.get("spark_env.spark_task2", auto_version="latest")


@env.task
async def root_task() -> float:
    print(f"Running torch_task: {torch_task}", flush=True)

    # Note that the input to the task is a dictionary, which is converted to the typed model
    v = await torch_task(v={"x": 12, "y": 13})
    print(f"Completed torch_task: {v}, {type(v)}", flush=True)

    # The output is a DataClass, which has attributes that can be accessed
    # you cannot subscript it like a dict, but you can access the attributes
    print(f"{v.some_string}, {v.some_float}, {v.some_bool}", flush=True)  # The type is faked in flyte.types

    print(f"Running spark_task: {spark_task}", flush=True)
    v2 = await spark_task(x="hello")
    print(f"Completed spark_task: {v2}, {type(v2)}", flush=True)
    print(f"{v2.feature_a}, {v2.feature_b}", flush=True)

    return await spark_task2(r=v2)


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.with_runcontext(labels={"x": "y"}, annotations={"x": "y"}).run(root_task)
    print(r.url)
