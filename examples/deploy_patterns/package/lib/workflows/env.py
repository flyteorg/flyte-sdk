import flyte

env = flyte.TaskEnvironment(
    name="folder-example",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)

